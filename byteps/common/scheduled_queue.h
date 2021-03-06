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

#ifndef BYTEPS_SCHEDULED_QUEUE_H
#define BYTEPS_SCHEDULED_QUEUE_H

#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>
#include "common.h"
#include "ready_table.h"

namespace byteps {
namespace common {

class BytePSScheduledQueue {
 public:
  BytePSScheduledQueue(QueueType type);
  QueueType getQueueType() { return _qt; }
  void addTask(std::shared_ptr<TensorTableEntry>);
  void recorderTs(std::shared_ptr<TensorTableEntry>);
  std::shared_ptr<TensorTableEntry> getTask();
  std::shared_ptr<TensorTableEntry> getTask(uint64_t key);
  uint32_t pendingSize();
  void reportFinish(std::shared_ptr<TensorTableEntry> task);
  void tune_bandwidth_by_weights(std::shared_ptr<TensorTableEntry> task);
  double weight;

 private:
  // TODO: use priority queue or heap
  std::vector<std::shared_ptr<TensorTableEntry>> _sq;
  std::mutex _mutex;
  uint64_t _credits;
  bool _is_scheduled;
  QueueType _qt;
  ReadyTable *_rt;
//chris paramter
  int _chris_tuning;
  int _chris_info;
  int _tuning_on=0;
  int _chris_bandwidth;
  int _old_bd_ps=0;
  int _old_bd_worker=0;
  double _chris_threshold;
  double _chris_pull_base;
  int _chris_dividend;
  std::string _worker_id;
  std::string _old_command;
  std::string _chris_network;
};

}  // namespace common
}  // namespace byteps

#endif  // BYTEPS_SCHEDULED_QUEUE_H
