# DeepBot-A-Deep-Learning-Approach-for-Botnet-Traffic-Detection
The Application of Deep Learning in Cybersecurity
Application of Deep Learning in Cybersecurity
1. Project Background and Motivation
With the explosive growth of IoT devices and the increasing sophistication of cyberattacks, Botnets have emerged as a significant threat to cybersecurity. By infecting and remotely controlling large numbers of devices, attackers can form distributed networks (botnets) to launch DDoS attacks, send spam, conduct data theft, or deploy ransomware, posing serious threats to systems and privacy.

Traditional rule-based intrusion detection systems often struggle against evolving attack patterns. Therefore, deep learning is increasingly applied to automatically learn patterns from network traffic features, enhancing detection accuracy and uncovering stealthy threats. This project explores and compares deep learning-based approaches for botnet traffic detection.

2. Dataset Exploration and Preprocessing
📌 Dataset Source
We use the publicly available CTU-13 dataset, provided by the Czech Technical University, which consists of NetFlow-based network traffic from 13 scenarios containing different types of botnets (e.g., Neris, Rbot, Sogou, Donbot).

Labeling: Binary classification – 0: Normal, 1: Botnet

Preprocessing Steps:

Merging 13 .binetflow files into a single dataset;

Extracting key features: Dur, Proto, Sport, Dport, TotBytes, TotPkts;

Encoding Proto, converting Sport/Dport to integers;

Applying StandardScaler for normalization;

Label mapping: Botnet → 1, others → 0;

Splitting into training/testing sets (80:20);

Saving processed data as PyTorch tensors.

✅ Script: merge_data.py produces merged_ctu13.csv

🔧 Feature Description (Partial)
Feature	Description
Dur	Flow duration
Proto	Protocol type (e.g., TCP, UDP)
Sport/Dport	Source/Destination port numbers
TotBytes	Total bytes transmitted
TotPkts	Total packets transmitted
Label	Binary indicator for Botnet activity

3. Model Development
✅ 1. CNN Architecture
Input Shape: (Batch, 1, 6)

Layers: Conv1d → BatchNorm → ReLU → Dropout → FC → Dropout → Softmax

Optimizer: Adam (lr=0.001)

Loss: CrossEntropy + class_weight balancing

EarlyStopping: Stop after 5 non-improving epochs

Batch Size: 8192

Mixed Precision Training: Enabled via torch.amp

▶️ Run train_cnn.py (epoch=50)
![image](https://github.com/user-attachments/assets/56eac5b0-7d95-47af-8a21-4b6a380e8ea2)

Figure 1: CNN Loss & Accuracy Curves


✅ 2. LSTM Architecture
Input Shape: (Batch, 6, 1)

Layers: LSTM → BatchNorm → Dropout → FC → Output

Suitable for capturing temporal dependencies in network flow data

Training configuration same as CNN

▶️ Run train_lstm.py (epoch=50)
![image](https://github.com/user-attachments/assets/a1dd73db-d44d-46c1-a054-29d69bc0a66f)

Figure 2: LSTM Loss & Accuracy Curves


4. Model Evaluation
📈 CNN Evaluation (evaluate_cnn.py)
![image](https://github.com/user-attachments/assets/023793f6-8e5f-4bf7-8cee-fe55f7a8c298)




📈 LSTM Evaluation (evaluate_lstm.py)
![image](https://github.com/user-attachments/assets/6603abc1-30fd-427a-83c0-5df791734782)




📊 Metric Comparison
Metric	CNN	LSTM
Accuracy	0.9396	0.9762 ✅
Precision	0.2184	0.4156 ✅
Recall	0.9946	0.9930
F1-score	0.3582	0.5860 ✅
ROC-AUC	0.9935	0.9985 ✅
Model Size	Small	Larger
Speed	Fast	Slower

✅ Conclusion: LSTM demonstrates better overall performance and lower false positive rate, though at the cost of longer training time.

5. Extended Discussion: CNN vs LSTM
Aspect	CNN	LSTM
Strengths	Simple, fast, edge-deployable	Powerful for time-dependent patterns
Weaknesses	Low precision, high false positives	Resource intensive
Best Use Cases	Real-time filtering, resource-constrained devices	Security-critical environments
Misclassification	High	Lower ✅

📌 Future Work
Explore Transformer architectures;

Integrate attention mechanism;

Expand to multi-class labels: Background / Normal / Botnet.

6. Conclusion & Takeaways
I independently completed the entire deep learning workflow for botnet detection, from data acquisition, preprocessing, feature engineering, to model design, training, tuning, and evaluation. I downloaded the CTU-13 dataset from https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/, merged multiple flow files, and applied encoding, normalization, and class balancing strategies.

Two deep learning models — CNN and LSTM — were implemented, each incorporating BatchNorm, Dropout, EarlyStopping, and AMP-based mixed precision training. Through comparative experiments, I gained practical insights into the strengths and trade-offs of different architectures in cybersecurity applications, particularly the superior F1-score and robustness of LSTM in detecting time-dependent botnet activity.

This project not only deepened my technical skills in deep learning for network security but also laid a solid foundation for future work in intelligent threat detection.
