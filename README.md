# Note on this fork
This is merely a fork for more convenient usage in a special case. I mostly added some logging and the rmt_laser.py script now takes 3 parameters which makes it a little easier to deploy in colab/kaggle/runpod:
rmt_laser.py {MODEL_ID} {username} {token}

# Original README text follows...

## Optimizing Large Language Models Using Layer-Selective Rank Reduction and Random Matrix Theory

## Abstract
In this study, we introduce a novel adaptation of Layer-Selective Rank Reduction (LASER) for optimizing large language models, utilizing Marchenko-Pastur law from Random Matrix Theory. This approach marks a key advancement from the brute-force search methodology proposed in the original LASER framework. Our method strategically reduces model complexity while preserving, or even enhancing, performance as measured by perplexity. This targeted reduction, guided by the mathematical principles of Marchenko-Pastur, results in a more efficient and effective optimization process, setting a new standard for language model refinement.

## Introduction
The burgeoning field of large language models (LLMs) has introduced a host of computational and efficiency challenges. As these models grow in size and complexity, optimizing their structure without compromising performance becomes crucial. This paper introduces an innovative adaptation to the Layer-Selective Rank Reduction (LASER) approach, integrating the Marchenko-Pastur law from Random Matrix Theory. This integration marks a significant departure from the brute-force search method in the original LASER framework. We propose a more efficient and mathematically grounded method to reduce the complexity of LLMs. This method not only maintains but potentially enhances the model's performance, as measured by perplexity. By leveraging the principles of Random Matrix Theory, our approach provides a systematic and theoretically robust framework for optimizing large-scale language models, highlighting the potential for more nuanced and effective model refinement strategies.
Our approach advances the LASER framework by strategically employing the Marchenko-Pastur law to identify and eliminate redundant components in LLM layers. This methodology not only streamlines the model but also enhances its interpretability and efficiency. By moving beyond the limitations of brute-force methods, we open new avenues for optimizing neural networks. Our work underscores the synergy between advanced mathematical theories and practical AI applications, setting a precedent for future developments in the field. This paper will detail our methodology, experiments, and the implications of our findings for the broader landscape of LLM optimization.

## Key Concepts
Our methodology is grounded in the intersection of advanced machine learning techniques and mathematical theory. We focus on two main components: Layer-Selective Rank Reduction (LASER) and the Marchenko-Pastur law from Random Matrix Theory.

LASER Framework Adaptation: The core of our approach involves adapting the LASER technique, originally designed for reducing the complexity of neural networks by selectively pruning the weights of a model's layers. We enhance this process by implementing a more targeted selection method based on our mathematical framework.

Marchenko-Pastur Law: The Marchenko-Pastur law is a pivotal concept from Random Matrix Theory, used to determine the distribution of eigenvalues in large random matrices. In the context of our work, it guides the identification of redundant components in the weight matrices of LLMs. By applying this law, we can precisely estimate which singular values in a matrix are statistically significant and which are due to noise, allowing for effective complexity reduction without loss of key information.

Integration of Concepts: The integration of these two concepts enables a more refined approach to model optimization. Unlike brute-force methods, our technique uses the Marchenko-Pastur law to systematically identify and eliminate less important components in the model's layers. This results in a more efficient optimization process, potentially enhancing the model's performance and interpretability.

## Cite As
Fernando Fernandes Neto, David Golchinfar and Eric Hartford. "Optimizing Large Language Models Using Layer-Selective Rank Reduction and Random Matrix Theory." 2024.
