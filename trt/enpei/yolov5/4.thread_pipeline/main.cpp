// 多线程示例
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

// 队列大小
int buffer_size = 10;

// 定义了一个全局变量代表队列
std::queue<int> buffer;

// 互斥锁保证生产者和消费者的不能同时访问缓冲区
std::mutex buffer_mutex;
// 条件变量: 未满
std::condition_variable not_full;
// 条件变量: 未空
std::condition_variable not_empty;

void produce() {
    // 生产者
    for (int i = 1; i <= 20; i++) {
        // 使用互斥锁
        std::unique_lock<std::mutex> lock(buffer_mutex);
        // 队列满时阻塞生产者
        not_full.wait(lock, [] { return buffer.size() < buffer_size; });
        // 生产产品
        buffer.push(i);
        std::cout << "生产 " << i << std::endl;
        // 唤醒消费者
        not_empty.notify_one();
    }
}
void consume() {
    // 消费者
    while (true) {
        // 使用互斥锁
        std::unique_lock<std::mutex> lock(buffer_mutex);
        // 队列空时阻塞消费者
        not_empty.wait(lock, [] { return !buffer.empty(); });
        // 消费产品
        int val = buffer.front();
        buffer.pop();
        std::cout << "消费 " << val << std::endl;
        // 唤醒生产者
        not_full.notify_one();
        if (val == 20) {
            break;
        }
    }
}

int main() {
    // 多线程
    std::thread producer(produce);
    std::thread consumer(consume);

    producer.join();
    consumer.join();
    return 0;
}

// g++ main.cpp -std=c++14 -pthread