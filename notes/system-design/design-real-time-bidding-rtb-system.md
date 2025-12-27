---
title: "Design Real-Time Bidding (RTB) System"
date: "2025-12-27"
tags: ["system-design", "adtech", "high-concurrency"]
related: []
slug: "design-real-time-bidding-rtb-system"
category: "system-design"
---

# Design Real-Time Bidding (RTB) System

## Summary
A Real-Time Bidding (RTB) system (specifically a **DSP - Demand Side Platform**) must process millions of requests per second (QPS) with a strict latency budget (typically < 100ms end-to-end, < 20-50ms internal processing) to bid on ad impressions. Key challenges are **Latency**, **Scale**, and **Budget Pacing**.

## Details

### 1. High-Level Architecture
The flow starts when a user visits a website (Publisher).
1.  **User -> SSP (Supply Side Platform)**: "I have an ad slot."
2.  **SSP -> Ad Exchange**: Broadcasts the request to multiple DSPs.
3.  **Ad Exchange -> DSP (Our System)**: "Bid request for User X, Site Y. You have 50ms."
4.  **DSP**: Decides to bid $0.50.
5.  **Ad Exchange**: Picks winner -> SSP -> User sees ad.

### 2. DSP Internal Architecture (The "Bidder")
To meet the 50ms SLA, the Bidder is a highly optimized, distributed system.

#### Core Components
-   **Front-End Gateway**: Terminates SSL, parses OpenRTB protocol, enforces timeouts. Fails fast if overloaded.
-   **User Profile Service**: Fast key-value store (Redis/Aerospike). Looks up User ID -> Segments (e.g., "Male", "Interested in Cars").
-   **Retrieval/Candidate Generation**: filters millions of ads down to ~500 candidates.
    -   *Technique*: Inverted Index (Targeting Criteria -> Ads). "Find ads targeting 'Cars' AND 'California'".
-   **Prediction Service (Inference)**: Scores the ~500 candidates using ML models (CTR * CVR).
    -   *Constraint*: Can't run heavy Deep Learning on all 500. Use a **Cascade**:
        -   Stage 1: Light model (Vector dot product) on 500 items.
        -   Stage 2: Heavy model (DeepFM) on top 50 items.
-   **Budget & Pacing Service**: Checks if the advertiser has budget.
    -   *Challenge*: Distributed counting. Updating a central database for every impression is too slow.
    -   *Solution*: **Probabilistic Throttling** or **Batch Sync**. Allocate slices of budget to individual bidder instances. Sync every minute.

### 3. Latency Optimization Techniques
-   **Timeouts**: Set strict internal timeouts (e.g., User Profile: 5ms, Inference: 20ms). If a service is slow, return "No Bid" immediately.
-   **Data Locality**: Keep User Profile data in memory or closest cache.
-   **Feature Store**: Pre-compute heavy features (e.g., "User's average spend last 30 days") and store in Redis. Don't compute SQL at runtime.

### 4. Data Flow & Logging
-   **Bid Logs**: Every bid decision must be logged for training data.
-   **Win Logs**: We only pay if we win. The Ad Exchange sends a "Win Notification" later.
-   **Join**: We must join Bid Logs (Features) with Win Logs (Labels) to create the training dataset.

## Examples / snippets

### Latency Budget Breakdown (Total 50ms)
```text
| Component          | Time Budget | Notes                                |
|--------------------|-------------|--------------------------------------|
| Network overhead   | 5ms         | JSON parsing, serialization          |
| User Profile Lookup| 5ms         | Redis/Aerospike GET                  |
| Retrieval          | 10ms        | Inverted index query                 |
| Scoring (Inference)| 20ms        | The heavy lifter (Batch inference)   |
| Budget/Pacing Check| 5ms         | In-memory check                      |
| Logic/Overhead     | 5ms         | Business rules, bid shading          |
| ------------------ | ---         |                                      |
| TOTAL              | 50ms        |                                      |
```

## Flashcards

- What is the primary constraint in an RTB system? ::: **Latency** (typically < 100ms total, < 50ms internal).
- What database type is best for User Profile lookups in RTB? ::: **Key-Value Stores** like Redis or Aerospike (low latency, high throughput).
- How do we handle heavy ML inference within 20ms? ::: Use a **Cascade** (light model for filtration, heavy model for ranking) or optimize model serving (quantization, compilation).
- What is "Pacing" in AdTech? ::: Spreading an advertiser's budget evenly throughout the day so they don't spend it all in the first hour.
- Why is distributed budget management difficult? ::: Because locking a central database for every bid is too slow; we use **local allocation** and **batch synchronization**.

## Quizzes

### Architecture Decisions
Q: Your RTB system is timing out. Profiling shows the User Profile Service (PostgreSQL) is taking 100ms on complex joins. How do you fix this?
Options:
- A) Add more indexes to PostgreSQL.
- B) Move User Profiles to a wide-column store or Key-Value store (Cassandra/Aerospike) and pre-compute segments.
- C) Increase the timeout to 200ms.
- D) Cache the entire database in Python memory.
Answers: B
Explanation: Relational joins are too slow for RTB. You must denormalize data. Pre-computing segments ("User X -> [Segment A, Segment B]") and storing in a high-performance KV store is the standard solution.

### Budget Handling
Q: You have 100 bidder instances. An advertiser has a $100 budget. How do you prevent overspending without slowing down bidding?
Options:
- A) Global Lock: Check a central Redis counter before every bid.
- B) Local Allocation: Give each bidder $1. When it runs out, it asks a central coordinator for another $1.
- C) Post-processing: Bid as much as you want, cancel ads later.
- D) Randomly bid on 1% of traffic.
Answers: B
Explanation: Local Allocation minimizes network calls. "Check-and-set" on a central store (A) introduces massive contention and latency. Option B ensures strict limits with minimal overhead.

## Learning Sources
- [High Scalability: Architecture of an Ad Server](http://highscalability.com/) - Search for "Ad Server Architecture".
- [Introduction to Real-Time Bidding (IAB)](https://www.iab.com/guidelines/real-time-bidding-rtb-project/) - Industry standard protocols (OpenRTB).
- [Aerospike for AdTech](https://aerospike.com/solutions/ad-tech/) - Why KV stores are used in this domain.
- [Building a Bidder (Technical Blog)](https://tech.adroll.com/blog/) - Engineering blogs from AdRoll, The Trade Desk, or Criteo.
