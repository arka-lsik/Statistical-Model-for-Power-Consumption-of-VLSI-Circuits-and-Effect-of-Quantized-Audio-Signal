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

To know for $\textcolor{green}{mathemtical \space foundation \space of \space this \space part}$ of the project kindly visit either the PDF attached as *MTP* [page -] or [Link](https://www.google.com).

*Results Discussion*
All the resullts are refering from uploaded $\textcolor{blue}{Results \space A-to-D}$ section here [Link](https://github.com/arka-lsik/Statistical-Model-for-Power-Consumption-of-VLSI-Circuits-and-Effect-of-Quantized-Audio-Signal/tree/main/Results%20A-to-D)
- This $\textcolor{red}{ two Picture}$ [Link1](https://github.com/arka-lsik/Statistical-Model-for-Power-Consumption-of-VLSI-Circuits-and-Effect-of-Quantized-Audio-Signal/blob/main/Synthetic%20Audio%20dataset/datasetpic1.png) & [Link2](https://github.com/arka-lsik/Statistical-Model-for-Power-Consumption-of-VLSI-Circuits-and-Effect-of-Quantized-Audio-Signal/blob/main/Synthetic%20Audio%20dataset/datasetpic2.png) represents the GMM signal as clustered form (top view) with thousand and ten thousand signal samples with the two different nature of dataset.
-  the impact of quantization at different levels was analyzed, revealing changes in the data values.
  - The $\textcolor{red}{Pic-1}$ [Link](https://github.com/arka-lsik/Statistical-Model-for-Power-Consumption-of-VLSI-Circuits-and-Effect-of-Quantized-Audio-Signal/blob/main/Results%20A-to-D/Pic-1.png) Probability Distribution Function estimated after Uniform Quantization with Audio Dataset1. Results showed Level of Quantization(L) with Quantization Support(n).
  - The $\textcolor{red}{Pic-2}$ [Link](https://github.com/arka-lsik/Statistical-Model-for-Power-Consumption-of-VLSI-Circuits-and-Effect-of-Quantized-Audio-Signal/blob/main/Results%20A-to-D/Pic-2.png) Probability Distribution Function estimated after Uniform Quantization with Audio Dataset2. Results showed Level of Quantization(L) with Quantization Support(n).
  - The $\textcolor{red}{Pic-2}$ [Link](https://github.com/arka-lsik/Statistical-Model-for-Power-Consumption-of-VLSI-Circuits-and-Effect-of-Quantized-Audio-Signal/blob/main/Results%20A-to-D/Pic-3.png) Probability Distribution Function estimated after Entropy-Coded Quantization with Audio Dataset1. Results showed Level of Quantization(L) with Quantization Support(n).
  - The $\textcolor{red}{Pic-2}$ [Link](https://github.com/arka-lsik/Statistical-Model-for-Power-Consumption-of-VLSI-Circuits-and-Effect-of-Quantized-Audio-Signal/blob/main/Results%20A-to-D/Pic-4.png) Probability Distribution Function estimated after Entropy-Coded Quantization with Audio Dataset2. Results showed Level of Quantization(L) with Quantization Support(n).
    - In the $\textcolor{red}{f2nd \space case}$ of optimal quantizer(ECQ - Vector Quantizer) I got the same data sample estimated with level of 8 of quantizer rather than 16 level. 
- These $\textcolor{red}{four \space Pictures \space}$ [Link1](https://github.com/arka-lsik/Statistical-Model-for-Power-Consumption-of-VLSI-Circuits-and-Effect-of-Quantized-Audio-Signal/blob/main/Results%20A-to-D/Pic-5.png), [Link1](https://github.com/arka-lsik/Statistical-Model-for-Power-Consumption-of-VLSI-Circuits-and-Effect-of-Quantized-Audio-Signal/blob/main/Results%20A-to-D/Pic-6.png), [Link1](https://github.com/arka-lsik/Statistical-Model-for-Power-Consumption-of-VLSI-Circuits-and-Effect-of-Quantized-Audio-Signal/blob/main/Results%20A-to-D/Pic-7.png), [Link1](https://github.com/arka-lsik/Statistical-Model-for-Power-Consumption-of-VLSI-Circuits-and-Effect-of-Quantized-Audio-Signal/blob/main/Results%20A-to-D/Pic-7.png) are suggesting the *Mean Square Error* results after quantization of GMM modeled signal, compared with distorted signal with the original one. Here I used as previously said the *Expectation Maximization* algorithm for better clustering. So comparison showed that for bosth case of dataset and for both uniform and optimal quantizer, the EM applied estimation noise is effectively reduced in significatant numbers than without EM one.

## 📝 **Details of Algorithm Pipeline for Power Estimation of VLSI Circuits**

![Workflow thinking process step by step](Power-Estimation-of-VLSI-circuits/PE.png)    

🎯**Particular problems I am foucusing**

With the advancements in $\textcolor{green}{VLSI \space technology}$, it is now possible to integrate millions of transistors into a single chip. However,increasing the count of transistors and operating frequencies leads to a significant rise in power consumption. This has emerged as a critical issue, especially in submicron technologies. Therefore, $\textcolor{yellow}{accurate \space power \space estimation \space during \space the \space early \space stages \space of \space VLSI \space circuit \space design \space is \space essential}$, as power consumption greatly affects the overall reliability and performance of the system and redesign time or power optimizing cost at RTL, Algorithmic or Architectural level.

🚀 **Data set I work for**

For power estimation I worked with my model on two dataset. The Simple one(first one) by visiting this [Link](https://github.com/arka-lsik/Statistical-Model-for-Power-Consumption-of-VLSI-Circuits-and-Effect-of-Quantized-Audio-Signal/blob/main/Power-Estimation-of-VLSI-circuits/power%20consumption%20of%20ISCAS89%20benchmark%20circuits.csv) you can observe.
- This dataset I taken form $\textcolor{red}{ISCAS'89}$ conference which contains of 20 different bench mark sequential circuits.
- The every benchmark sequential logic is made with various logic gates as a load like NOR, NAND, NOT etc. So total 9 input features.
- At end column its given the Mc(monte carlo) in mw simulted power. I have to estimate as close as possible.

As this datset [Link]() is Old and with small quantites data values there may be a industry limitation to judge this model perfomance. 
- So next I found the another dataset form $\textcolor{red}{PATMOS'17}$ conference.
- Consider Odorid-XU3 for this work, which contains an Exynos-5422 SoC with a quad-core ARM Cortex-A7 CPU cluster and a quad-core ARM Cortex-A15 CPU cluster.
- Both types of CPU share the same instruction-set architecture (ISA) but are optimised for different energy/performance design.
- The power measurements data (in the sheet represented) were collected from power sensors built into the ODROID-XU3 platform.
- This work data considers only the higher performance Cortex-A15 cluster.

🎯 **Model Mathematical Descriptions**

The core idea is to model the joint distribution of input features $\textcolor{green}{(X)}$ and target variable $\textcolor{green}{(Y)}$ using a mixture of Gaussian distributions, and then use this joint model to predict $\textcolor{green}{Y \space given \space X}$. Let's break down the mathematics for 1st case of dataset- 

*1. The Regression Problem Setup*
- Features (X): These are input variables (e.g., GATE, AND, INV counts). X is an N×D matrix, where N is the number of data points and D=9 is the number of features.
- Target (Y): This is the variable I want to predict (Power Consumption). Y is an N×1 vector.
- Goal: To learn a function f(X)→Y such that for a new circuit with features x, we can predict its power consumption y.

*2. Data Preprocessing: Normalization (Z-score Standardization)*
- Purpose:
- -Scale Invariance: Prevents features with larger numerical ranges from disproportionately influencing the model. If 'GATE' count ranges from 10 to 1000 and 'INV' count from 1 to 10, 'GATE' would dominate calculations without scaling.
- -Improved Convergence for EM: Many iterative optimization algorithms (like EM for GMMs) converge faster and more reliably when features are on a similar scale. So this a necessary steps.
- For each feature (column) in X and for Y:
  The equations are:

$$
x'_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

$$
y'_i = \frac{y_i - \mu_Y}{\sigma_Y}
$$



