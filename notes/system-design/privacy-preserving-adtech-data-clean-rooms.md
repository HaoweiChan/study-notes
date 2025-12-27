---
title: "Privacy-Preserving AdTech (Data Clean Rooms)"
date: "2025-12-27"
tags: ["privacy", "adtech", "system-design"]
related: []
slug: "privacy-preserving-adtech-data-clean-rooms"
category: "system-design"
---

# Privacy-Preserving AdTech (Data Clean Rooms)

## Summary
With the deprecation of 3rd-party cookies (Chrome, Safari ITP) and stricter regulations (GDPR, CCPA), AdTech is shifting from tracking users across sites to privacy-first architectures. Key solutions include **Data Clean Rooms (DCRs)**, **Federated Learning**, and **Differential Privacy**.

## Details

### 1. The Context: The "Cookiepocalypse"
- **Old World**: AdTech vendors placed 3rd-party cookies on Publisher sites to track users everywhere.
- **New World**: Browsers block 3rd-party cookies. IDFA (iOS) requires opt-in.
- **Impact**: Attribution ("Did this ad cause a sale?") and Targeting ("Find users who like shoes") are broken.

### 2. Data Clean Rooms (DCR)
- **Concept**: A secure environment where two parties (e.g., a Retailer and a Publisher) can join their 1st-party data to calculate aggregate insights *without* ever revealing raw user rows to each other.
- **Mechanism**:
    1.  **Ingestion**: Party A uploads hashed emails (SHA256). Party B uploads hashed emails.
    2.  **Matching**: The DCR finds the intersection (Common IDs).
    3.  **Aggregation**: The DCR runs a script (SQL) to calculate "Total Spend of Matched Users".
    4.  **Output**: Only the aggregate number is returned.
- **Providers**: Snowflake, AWS Clean Rooms, InfoSum, LiveRamp.

### 3. Federated Learning (FL)
- **Concept**: "Bring the model to the data, not the data to the model."
- **Mechanism**:
    1.  A global model is sent to user devices (Edge).
    2.  The device trains the model locally on private data (e.g., browsing history).
    3.  Only the **Model Gradients** (updates) are sent back to the central server.
    4.  The server aggregates gradients to update the global model.
-   **Use Case**: Google Keyboard (Gboard) prediction, Google Privacy Sandbox (Topics API).

### 4. Differential Privacy (DP)
- **Concept**: Adding mathematical noise to queries so that the output does not reveal whether any *single individual* was present in the dataset.
-   **Mechanism**: Add Laplacian noise to the count. $Count_{reported} = Count_{true} + Noise$.
-   **Privacy Budget ($\epsilon$)**: Measures how much privacy is leaked.

## Examples / snippets

### Differential Privacy Concept (Python)

```python
import numpy as np

def differentially_private_sum(data, epsilon):
    """
    Returns the sum of data with Laplace noise for privacy.
    epsilon: Privacy budget (lower is more private)
    sensitivity: Max change one individual can make (e.g., 1 for count)
    """
    true_sum = np.sum(data)
    sensitivity = 1.0 
    
    # Scale of noise = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=sensitivity/epsilon)
    
    return true_sum + noise

# Example: 100 users, 50 clicked.
clicks = [1] * 50 + [0] * 50
print(f"True Sum: {sum(clicks)}")
print(f"DP Sum (e=0.1): {differentially_private_sum(clicks, 0.1):.2f}")
# Output might be 52.3 or 41.5. Hide the exact '50'.
```

## Flashcards

- What is a Data Clean Room (DCR)? ::: A secure environment allowing two parties to join and analyze their first-party data (like email lists) to gain aggregate insights without revealing raw customer data to each other.
- What is the primary goal of Federated Learning? ::: To train machine learning models on decentralized data (e.g., user devices) without ever uploading the private raw data to a central server.
- What is Differential Privacy? ::: A mathematical definition of privacy that adds noise to query results to ensure the output doesn't reveal the presence or absence of any single individual.
- Why are 3rd-party cookies being deprecated? ::: To protect user privacy by preventing cross-site tracking and profiling by unconsented AdTech vendors.
- What is "Sensitivity" in Differential Privacy? ::: The maximum amount that the output of a function can change if a single individual's data is added or removed.

## Quizzes

### DCR Usage
Q: A Retailer wants to know if their ads on a News Site led to purchases. They both have user email addresses. Why use a Data Clean Room instead of emailing the CSVs?
Options:
- A) DCRs are faster.
- B) Sending raw user lists violates privacy laws (GDPR) and exposes business secrets (customer lists). DCRs calculate the overlap count/value without sharing the lists.
- C) CSVs are too big.
- D) DCRs use AI.
Answers: B
Explanation: Sharing raw PII (Personally Identifiable Information) like email lists is a massive privacy risk and compliance violation. DCRs allow the mathematical operation (Intersection) without the data exposure.

### Federated Learning
Q: In Federated Learning, what is sent from the user's device to the central server?
Options:
- A) The user's browsing history.
- B) The user's raw input text.
- C) Model weight updates (Gradients).
- D) An encrypted copy of the hard drive.
Answers: C
Explanation: The raw data stays on the device. The local model computes gradients (how to improve the model based on local data), and only these mathematical updates are sent to the cloud.

## Learning Sources
- [Snowflake Data Clean Rooms](https://www.snowflake.com/en/data-cloud/workloads/data-collaboration/) - Industry standard DCR.
- [Google Privacy Sandbox: Topics API](https://privacysandbox.com/proposals/topics/) - Replacement for cookies.
- [Federated Learning: Comic Guide (Google)](https://federated.withgoogle.com/) - Visual introduction.
- [Differential Privacy for Dummies](https://desfontaines.medium.com/differential-privacy-for-dummies-a-non-technical-explanation-4c539b341d48) - High-level concept.
