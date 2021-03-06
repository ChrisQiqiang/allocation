// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "scheduled_queue.h"
#include <algorithm>
#include <cmath>
#include "global.h"
#include "logging.h"

namespace byteps {
namespace common {

BytePSScheduledQueue::BytePSScheduledQueue(QueueType type) {
  if (type == REDUCE && BytePSGlobal::GetNccl()->IsSignalRoot()) {
    _is_scheduled = true;
  } else {
    _is_scheduled = false;
  }

  size_t credit_in_partition = BytePSGlobal::GetNccl()->GetGroupSize() + 1;

  auto byteps_scheduling_credit = getenv("BYTEPS_SCHEDULING_CREDIT");
  credit_in_partition = byteps_scheduling_credit ? atoi(byteps_scheduling_credit) : 0;
  if (!credit_in_partition) { // disable scheduling by default
    _is_scheduled = false;
  }
  weight=0;
  _qt = type;
  _credits = _is_scheduled
              ? BytePSGlobal::GetPartitionBound() * credit_in_partition
              : 34359738368;  // 32GB, basically disabling credit control

  auto tuning = getenv("CHRIS_TUNING");
  _chris_tuning = tuning ? atoi(tuning) : 0;
  auto info = getenv("CHRIS_INFO");
  _chris_info = info ? atoi(info) : 0;
  auto maxbandwidth = getenv("CHRIS_MAX_BANDWIDTH");
  _chris_bandwidth = maxbandwidth ? atoi(maxbandwidth) : INT_MAX;
  auto threshold = getenv("CHRIS_THRESHOLD");
  _chris_threshold = threshold ? double(atoi(threshold)) / 100 : 0.1;
  _worker_id = getenv("DMLC_WORKER_ID");
  auto _pull_base = getenv("CHRIS_PULL_BASE");
  _chris_pull_base = _pull_base ? double(atoi(_pull_base)) / 100 : 0.1;
  auto _dividend = getenv("CHRIS_DIVIDEND");
  _chris_dividend = _dividend ? atoi(_dividend) : 10;
  // if(_chris_network == "vgg")
  //   _dividend = 5;
  if(BytePSGlobal::IsRootDevice() && _qt == PULL)
    BPS_LOG(INFO) << "_chris_tuning:" << _chris_tuning << "   _chris_info: " << _chris_info 
                << "  _chris_bandwidth: " << _chris_bandwidth << "  _chris_threshold:" << _chris_threshold 
                << " worker id:" << _worker_id << " _dividend:" << _dividend;
  
  if(BytePSGlobal::IsRootDevice() && _qt == PUSH){
      if(_chris_tuning){
        std::string tc_command;
        if(_chris_tuning == 10)
          tc_command = "sudo sh tc_init.sh -l 0";
        else
          tc_command = "sudo sh tc_init.sh -l 1";
        BPS_LOG(INFO) << "task bandwidth limit done";
        system(tc_command.c_str());  
      }
  }

  _rt = nullptr;

  switch (_qt) {
    case REDUCE:
      if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
        _rt = BytePSGlobal::GetReduceTable();
      }
      break;
    case PCIE_REDUCE:
      if (BytePSGlobal::IsCrossPcieSwitch()) {
        if (BytePSGlobal::GetCpuReducer()->isRoot()) {
          _rt = BytePSGlobal::GetPcieReduceTable();
        }
      }
      break;
    case PUSH:
      if (BytePSGlobal::IsRootDevice()) {
        _rt = BytePSGlobal::GetPushTable();
      }
      break;
    case COPYH2D:
      if (!BytePSGlobal::IsRootDevice()) {
        _rt = BytePSGlobal::GetCopyTable();
      }
      break;
    case BROADCAST:
      if (BytePSGlobal::GetNccl()->IsSignalRoot()) {
        _rt = BytePSGlobal::GetBroadcastTable();
      }
      break;
    default:
      break;
  }
}

void BytePSScheduledQueue::addTask(std::shared_ptr<TensorTableEntry> entry) {
  std::lock_guard<std::mutex> lock(_mutex);
  _sq.push_back(entry);
  if (_is_scheduled) {
    // TODO: below can be optimized to O(n) using insertion sort
    std::sort(
        _sq.begin(), _sq.end(),
        [](std::shared_ptr<TensorTableEntry> a,
           std::shared_ptr<TensorTableEntry> b) {
          if (a->priority == b->priority) {
            return (a->key < b->key);  // from the first partition to the last
          }
          return (a->priority > b->priority);  // from higher priority to lower
        });
  }
  BPS_CHECK(entry->tensor_name != "");
  if(!_tuning_on && entry -> priority != 0)_tuning_on = 1;
  BPS_LOG(TRACE) << "Queue " << LogStrings[_qt]
                 << " addTask: " << entry->tensor_name << " key: " << entry->key
                 << " rank: " << BytePSGlobal::GetLocalRank();
  return;
}

// Record the start time of the sub-tasks for all QueueTypes of each partition.
void BytePSScheduledQueue::recorderTs(std::shared_ptr<TensorTableEntry> task) {
  auto context = task->context;
  // add for profiling
  if (context->profile_flag) {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(duration);

    auto &queue_list = task->queue_list;
    BPS_CHECK_GE(queue_list.size(), 1);
    auto this_op = queue_list[0];

    BPSCommTime *ret = new BPSCommTime;
    ret->start_t = (long long)(us.count());
    ret->key = task->key;
    ret->type = this_op;
    context->part_comm_time[task->key][this_op].push(ret);
  }
}

std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::getTask() {
  std::lock_guard<std::mutex> lock(_mutex);
  std::shared_ptr<TensorTableEntry> task;
  // TODO: below can be optimized -- if we take task from the tail, erase() can
  // be faster
  for (auto it = _sq.begin(); it != _sq.end(); ++it) {
    if ((*it)->ready_event) {
      if (!(*it)->ready_event->Ready()) {
        continue;
      }
    }
    if (_is_scheduled) {
      if ((*it)->len > _credits) {
        continue;
      }
    }
    if (_rt) {
      if (!_rt->IsKeyReady((*it)->key)) {
        continue;
      }
      _rt->ClearReadyCount((*it)->key);
    }
    task = *it;
    _sq.erase(it);
    if (_is_scheduled) {
      _credits -= task->len;
    }
    if(BytePSGlobal::IsRootDevice() && (_qt == PUSH || _qt == PULL) && _tuning_on){
      // weight += 100000 / ((task -> priority - 1) * (task -> priority - 1)); //func 1, 10000 / x^2 ,not ideal
      weight += _chris_bandwidth * exp(task -> priority / _chris_dividend); //func2, _chris_bandwidth * e ^ (-x / 10); aggregation
      if(_chris_info == 1)
        BPS_LOG(INFO) << "get task "  << LogStrings[_qt]  << task -> tensor_name 
                      << "  the priority is:" << task -> priority
                      << " weight:" << weight;
       // avoid duplicated adjustment
        tune_bandwidth_by_weights(task);
    }   
    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(TRACE) << "Queue " << LogStrings[_qt]
                   << " getTask: " << task->tensor_name << " key: " << task->key
                   << " rank: " << BytePSGlobal::GetLocalRank();
    task->ready_event = nullptr;
    // Add for profiling communication traces
    recorderTs(task);
    return task;
  }
  return nullptr;
}

void BytePSScheduledQueue::tune_bandwidth_by_weights(std::shared_ptr<TensorTableEntry> task){
    if(_qt == PUSH && BytePSGlobal::IsRootDevice()){
      auto pull_queue = BytePSGlobal::GetScheduledQueue(PULL);
      double pull_weight = pull_queue -> weight;
      std::string ps, worker;
      if(weight < _chris_bandwidth * exp(-1) || weight / (weight + pull_weight) < _chris_threshold){
        _old_bd_ps = _chris_bandwidth * 1;
        _old_bd_worker = _chris_bandwidth * 1;
      }
      else{
        _old_bd_ps = _chris_bandwidth * _chris_pull_base;
        _old_bd_worker = _chris_bandwidth * 1;
      }
      ps = std::to_string(_old_bd_ps);
      worker = std::to_string(_old_bd_worker);
      std::string  command = "sudo tc class change dev ens3 parent 1: classid 1:3 htb rate " + ps + "mbit \n sudo tc class change dev ens3 parent 1: classid 1:4 htb rate " + worker + "mbit";
      if(command == _old_command)
        return;
      if(_chris_tuning == 11 && _chris_info){
        BPS_LOG(INFO) << "worker " << _worker_id << " BANDWIDTH ALLOCATION BETWEEN PS TASK AND WORKER TASK."
               << "ps upload:" << ps <<  "  worker upload:" << worker
               << " push weight is:" << weight << " pull weight is:" << pull_weight;
      }
      if(_chris_tuning == 11){
        system(command.c_str());
        _old_command = command;
      }
        
    }   
  }


std::shared_ptr<TensorTableEntry> BytePSScheduledQueue::getTask(uint64_t key) {
  BPS_CHECK(!_is_scheduled);
  std::lock_guard<std::mutex> lock(_mutex);
  std::shared_ptr<TensorTableEntry> task;
  for (auto it = _sq.begin(); it != _sq.end(); ++it) {
    if ((*it)->ready_event) {
      BPS_CHECK((*it)->ready_event->Ready());
    }
    if ((*it)->key != (uint64_t)key) {
      continue;
    }
    task = *it;
    _sq.erase(it);

    BPS_CHECK(task->tensor_name != "");
    BPS_LOG(TRACE) << "Queue " << LogStrings[_qt]
                   << " getTask(key): " << task->tensor_name
                   << " key: " << task->key
                   << " rank: " << BytePSGlobal::GetLocalRank();
    task->ready_event = nullptr;
    // Add for profiling communication traces
    recorderTs(task);
    return task;
  }
  return nullptr;
}

uint32_t BytePSScheduledQueue::pendingSize() {
  std::lock_guard<std::mutex> lock(_mutex);
  return _sq.size();
}

void BytePSScheduledQueue::reportFinish(std::shared_ptr<TensorTableEntry> task) {
  if (_is_scheduled) {
    std::lock_guard<std::mutex> lock(_mutex);
    _credits += task -> len;
  }
  if((_qt == PUSH || _qt == PULL) && _tuning_on){
    std::lock_guard<std::mutex> lock(_mutex);
    // weight -= (100000 / ((task -> priority -1 ) * (task -> priority - 1)));
    weight -= _chris_bandwidth * exp(task -> priority / _chris_dividend); 
    if(_chris_info == 1)
        BPS_LOG(INFO) << "task finished reported: " << task -> tensor_name << "  the priority is:" << task -> priority;
      tune_bandwidth_by_weights(task);  // add
  } 
  return;
}

}  // namespace common
}  // namespace byteps
