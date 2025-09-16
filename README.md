# Machine Learning-based-Statistical-Model-for-Power-Consumption-of-VLSI-Circuits and Effect-of-Quantized-Audio-Signal

## 📚 **Abstarct**

This study explores how the noise came and average distortion is generate at detection end of practical end-to-end communication systems. The next part of this study explores where noise is not only introduced at the signal receiving end but also can accumulate at intermediate stages, particularly at quantization, source encoding \& channel encoding etc. Now it reflects the effect of data compression required by transmission over bandwidth-limited channels. In this context, I modeled the big data using a Gaussian mixture model (GMM), which undergoes uniform and entropy-coded quantization before clustering through the expectation-maximization (EM) algorithm. The analysis explores how clustering accuracy is affected by the number of quantization levels. This effective unsupervised machine learning algorithm, based on distribution-based clustering, can also address challenges in the VLSI design flow domain (including CPU/GPU design), particularly in estimating and optimizing power consumption of VLSI circuits. The approach is modeled through statistical predictive analysis and evaluated using several performance metrics for comprehensive comparison. Overall, this study provides a framework that connects the solution of signal processing problems to VLSI circuit domain challenges through the use of mathematical modeling.

→  **Keyword** : GMM(Gaussian Mixture Model), Noise, Quantization, EM. Power, Estimation

🎯 **Problem Statement**

● *Data Compression and Quantization in Bandwidth-Limited Channels*:
- When transmitting random source samples (e.g., Gaussian or mixture of Gaussians) over BW-limited channels, effective data compression is crucial to reduce transmission cost.
- Quantization is a necessary step in compression, but it inherently introduces information loss, reducing the accuracy of the original data representation.
- This becomes critical when quantized data is further used for downstream tasks such as clustering, where the distortion can significantly affect the final results.
- The Expectation-Maximization (EM) algorithm has been found to help mitigate the negative effects of quantization in clustering tasks, improving the robustness of data representation.

● *Research Shift Towards VLSI Circuit Analysis for Power-Aware Design*:
- Initial focus was on Audio signal processing and quantization impact reduction through clustering-based pipelines.
- Current research direction shifted toward a more impactful area: Circuit Analysis in VLSI Design.
- The main goal is to estimate and model pre-microarchitectural data to support power-aware design.
- This approach aims to reduce costly redesigns and improve timing efficiency by providing accurate early-stage power and performance estimations.

🚀 **Framework**
- Part 1: Is to choose the proper quantizer and its effect of quantization on GMM-based data model and applying EM algorithm on it for better clustering of recorded audio from different environment.
- Part 2: That’s GMM (for better result conditional GMM) and EM algorithm have a better impact of clustering and prediction on ”ISCAS’89 sequential Benchmark circuits”
data & "PATMOS'17"  quad-core ARM Cortex A15 CPU data for differernt events of activity.

## 📝 **Details of Algorithm Pipeline for Audio processing with effect of Quantization**

![Workflow thinking process step by step](Results%20A-to-D/workflow.png)

By looking this whole block wise diagram, I can simply say that the $\textcolor{green}{1st \space part}$ where if you carefully observe this is a end to end communication block. In that phase we generally talk about noise(Gaussian Noise) when we detect or estimate the signal at recevier end. But our focus of primary work shifted to the $\textcolor{yellow}{Analog \space to \space Digital \space Converter \space part \space for \space signal \space processing \space application}$ There Quantization Block which can contribute 7-8 % of noise in whole system. 

Now the $\textcolor{green}{2nd \space part}$ will be detailed analysis over Quantiation efect with different qunatization levels(L) for a complex Gaussian Mixture Modeled base digital signal. Where we done the experiments on synthetically generated Audio signal data passed through the one simple Uniform Quantizer and one iterative Entropy-Coded Quantizer. The after quantiation loss of originality of signal analyzed through estimation of signal with Expectation-Maximization iterative algorithm. Performnce evalated with MSE & MAE evaluation metrices.

$\textcolor{blue}{To \space know \space for \space mathemtical \space foundation \sace of \space this \space part \space of \space this \space project \space kindly \space visit \space either \space the \space PDF \space attached \space *MTP* \space or $ [Link](https://www.google.com)}.

## 📝 **Details of Algorithm Pipeline for Power Estimation of VLSI Circuits**
flowchart TD
    A[📥 Step 1: Data Preprocessing & Feature Engineering]
    A1[📂 Load Dataset: Dataset_PATMOS_2017.xlsx / Fig.4&5.csv]
    A2[⚙️ Select Features]
    A3[🎯 Target: Actual Power Consumption]
    A4[🧹 Data Cleaning (Handle missing values, outliers)]
    A5[📊 Feature Scaling (StandardScaler)]

    B[➗ Step 2: Data Splitting]
    B1[Split Data: 80% Training, 20% Testing]

    C[🏋️‍♀️ Step 3: Model Training]
    C1[📉 Baseline Model: Linear Regression]
    C2[⚡ Advanced Models: SVR, Random Forest, Simple Neural Network]
    C3[🔀 GMM-EM Hybrid Model]
    C3a[📊 Cluster Workloads using GMM-EM]
    C3b[📈 Train Separate Regression Model per Cluster]

    D[📊 Step 4: Model Evaluation]
    D1[📏 Metrics: MAPE, MAE, RMSE, R²]
    D2[📊 Compare Models’ Performance]

    E[⚙️ Step 5: Hyperparameter Tuning]
    E1[🔍 Grid Search + Cross-Validation]
    E2[🎯 Best Model Selection (Final Evaluation)]

    A --> A1 --> A2 --> A3 --> A4 --> A5 --> B
    B --> B1 --> C
    C --> C1
    C --> C2
    C --> C3 --> C3a --> C3b --> D
    D --> D1 --> D2 --> E
    E --> E1 --> E2
