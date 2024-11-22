<div align="center">
  <h1>🛠️ Awesome LMs with Tools</h1>
  <a href="https://awesome.re">
    <img src="https://awesome.re/badge.svg" alt="Awesome">
  </a>
  <a href="https://img.shields.io/badge/PRs-Welcome-red">
    <img src="https://img.shields.io/badge/PRs-Welcome-yellow" alt="PRs Welcome">
  </a>
  <a href="https://arxiv.org/abs/2403.15452">
    <img src="https://img.shields.io/badge/arXiv-2403.15452-b31b1b.svg" alt="arXiv">
  </a>
</div>

Language models (LMs) are powerful yet mostly for text-generation tasks. Tools have substantially enhanced their performance for tasks that require complex skills.

Based on our recent survey about LM-used tools, ["What Are Tools Anyway? A Survey from the Language Model Perspective"](https://arxiv.org/pdf/2403.15452), we provide a structured list of literature relevant to tool-augmented LMs.

- Tool basics ($\S2$)
- Tool use paradigm ($\S3$)
- Scenarios ($\S4$)
- Advanced methods ($\S5$)
- Evaluation ($\S6$)

If you find our paper or code useful, please cite the paper:

```bibtex
@article{wang2022what,
  title={What Are Tools Anyway? A Survey from the Language Model Perspective},
  author={Zhiruo Wang, Zhoujun Cheng, Hao Zhu, Daniel Fried, Graham Neubig},
  journal={arXiv preprint arXiv:2403.15452},
  year={2024}
}
``````

## $\S2$ Tool Basics

### $\S2.1$ What are tools? 🛠️
-  Definition and discussion of animal-used tools

   **Animal tool behavior: the use and manufacture of tools by animals** *Shumaker, Robert W., Kristina R. Walkup, and Benjamin B. Beck.* 2011 [[Book](https://books.google.com/books?hl=en&lr=&id=Dx7slq__udwC&oi=fnd&pg=PT1&dq=Animal+tool+behavior:+the+use+and+manufacture+of+tools+by+animals&ots=Wf6GmSG4uI&sig=48hv2QSipGyuCcucX-GnSJHscn8#v=onepage&q=Animal%20tool%20behavior%3A%20the%20use%20and%20manufacture%20of%20tools%20by%20animals&f=false)]

-  Early discussions on LM-used tools

   **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs** *Qin, Yujia, et al.* 2023.07 [[Paper]](https://openreview.net/forum?id=dHng2O0Jjr)

- A survey on augmented LMs, including tool augmentation
  
  **Augmented Language Models: a Survey** *Mialon, Grégoire, et al.* 2023.02 [[Paper]](https://openreview.net/forum?id=jh7wH2AzKK)

### $\S2.3$ Tools and "Agents" 🤖
- Definition of agents
  
  **Artificial intelligence a modern approach** *Russell, Stuart J., and Peter Norvig.* 2016 [[Book]](https://thuvienso.hoasen.edu.vn/handle/123456789/8967)

- Survey about agents that perceive and act in the environment
  
  **The Rise and Potential of Large Language Model Based Agents: A Survey** *Xi, Zhiheng, et al.* 2023.09 [[Preprint]](https://arxiv.org/abs/2309.07864)

- Survey about the cognitive architectures for language agents

  **Cognitive Architectures for Language Agents** *Sumers, Theodore R., et al.* 2023.09 [[Paper]](https://openreview.net/forum?id=1i6ZCvflQJ)

## $\S3$ The basic tool use paradigm

- Early works that set up the commonly used tooling paradigm
  
  **Toolformer: Language Models Can Teach Themselves to Use Tools** *Schick, Timo, et al.* 2024 [[Paper]](https://openreview.net/forum?id=Yacmpz84TH&referrer=%5Bthe%20profile%20of%20Roberto%20Dessi%5D(%2Fprofile%3Fid%3D~Roberto_Dessi1))

### Inference-time prompting

- Provide in-context examples for tool-using on visual programming problems
  
  **Visual Programming: Compositional visual reasoning without training** *Gupta, Tanmay, and Aniruddha Kembhavi.* 2023 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.pdf)

- Tool learning via in-context examples on reasoning problems involving text or multi-modal inputs
  
  **Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models** *Lu, Pan, et al.* 2024 [[Paper]](https://openreview.net/forum?id=HtqnVSCj3q&referrer=%5Bthe%20profile%20of%20Pan%20Lu%5D(%2Fprofile%3Fid%3D~Pan_Lu2))

- In-context learning based tool using for reasoning problems in BigBench and MMLU
  
  **ART: Automatic multi-step reasoning and tool-use for large language models** *Paranjape, Bhargavi, et al.* 2023.03 [[Preprint]](https://arxiv.org/abs/2303.09014)

- Providing tool documentation for in-context tool learning
  
  **Tool Documentation Enables Zero-Shot Tool-Usage with Large Language Models** *Hsieh, Cheng-Yu, et al.* 2023.08 [[Preprint]](https://arxiv.org/abs/2308.00675)

### Learning by training

- Training on human annotated examples of (NL input, tool-using solution output) pairs
  
  **API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs** *Li, Minghao, et al.* 2023.12 [[Paper]](https://aclanthology.org/2023.emnlp-main.187/)
  
  **Calc-X and Calcformers: Empowering Arithmetical Chain-of-Thought through Interaction with Symbolic Systems** *Kadlčík, Marek, et al.* 2023 [[Paper]](https://aclanthology.org/2023.emnlp-main.742.pdf)
  
- Training on model-synthesized examples
  
  **ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases** *Tang, Qiaoyu, et al.* 2023.06 [[Preprint]](https://arxiv.org/abs/2306.05301)
  
  **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs** *Qin, Yujia, et al.* 2023.07 [[Paper]](https://openreview.net/forum?id=dHng2O0Jjr)
  
  **MetaTool Benchmark for Large Language Models: Deciding Whether to Use Tools and Which to Use** *Huang, Yue, et al.* 2023.10 [[Paper]](https://openreview.net/forum?id=R0c2qtalgG&referrer=%5Bthe%20profile%20of%20Neil%20Zhenqiang%20Gong%5D(%2Fprofile%3Fid%3D~Neil_Zhenqiang_Gong1))

  **Making Language Models Better Tool Learners with Execution Feedback** *Qiao, Shuofei, et al.* 2023.05 [[Preprint]](https://arxiv.org/abs/2305.13068)
  
  **LLMs in the Imaginarium: Tool Learning through Simulated Trial and Error** *Wang, Boshi, et al.* 2024.03 [[Preprint]](https://arxiv.org/abs/2403.04746)

- Self-training with bootstrapped examples
  
  **Toolformer: Language Models Can Teach Themselves to Use Tools** *Schick, Timo, et al.* 2024 [Paper](https://openreview.net/forum?id=Yacmpz84TH&referrer=%5Bthe%20profile%20of%20Roberto%20Dessi%5D(%2Fprofile%3Fid%3D~Roberto_Dessi1))

## $\S4$ Scenarios

### Knowledge access 📚

- Collect data from structured knowledge sources, e.g., databases, knowledge graphs, etc.
  
  **LaMDA: Language Models for Dialog Applications** *Thoppilan, Romal, et al.* 2022.01 [[Paper]](https://arxiv.org/abs/2201.08239)
  
  **TALM: Tool Augmented Language Models** *Parisi, Aaron, Yao Zhao, and Noah Fiedel.* 2022.05 [[Preprint]](https://arxiv.org/abs/2205.12255)
  
  **ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings** *Hao, Shibo, et al.* 2024 [[Paper]](https://openreview.net/forum?id=BHXsb69bSx)
  
  **ToolQA: A Dataset for LLM Question Answering with External Tools** *Zhuang, Yuchen, et al.* 2024 [[Paper]](https://openreview.net/forum?id=pV1xV2RK6I)

  **Middleware for LLMs: Tools are Instrumental for Language Agents in Complex Environments** *Gu, Yu, et al.* 2024 [[Paper]](https://arxiv.org/abs/2402.14672)

  **GeneGPT: Augmenting Large Language Models with Domain Tools for Improved Access to Biomedical Information** *Jin, Qiao, et al.* 2024 [[Paper]](https://academic.oup.com/bioinformatics/article/40/2/btae075/7606338)

- Search information from the web
  
  **Internet-augmented language models through few-shot prompting for open-domain question answering** *Lazaridou, Angeliki, et al.* 2022.03 [[Paper]](https://arxiv.org/abs/2203.05115)
  
  **Internet-Augmented Dialogue Generation** *Komeili, Mojtaba, Kurt Shuster, and Jason Weston.* 2022 [[Paper]](https://aclanthology.org/2022.acl-long.579/)

- Viewing retrieval models as tools under the retrieval-augmented generation context
  
  **Retrieval-based Language Models and Applications** *Asai, Akari, et al.* 2023 [[Tutorial]](https://aclanthology.org/2023.acl-tutorials.6/)
  
  **Augmented Language Models: a Survey** *Mialon, Grégoire, et al.* 2023.02 [[Paper]](https://openreview.net/forum?id=jh7wH2AzKK)

### Computation activities 🔣

- Using calculator for math calculations
  
  **Toolformer: Language Models Can Teach Themselves to Use Tools** *Schick, Timo, et al.* 2024 [[Paper]](https://openreview.net/forum?id=Yacmpz84TH&referrer=%5Bthe%20profile%20of%20Roberto%20Dessi%5D(%2Fprofile%3Fid%3D~Roberto_Dessi1))

  **Calc-X and Calcformers: Empowering Arithmetical Chain-of-Thought through Interaction with Symbolic Systems** *Kadlčík, Marek, et al.* 2023 [[Paper]](https://aclanthology.org/2023.emnlp-main.742.pdf)

- Using programs/Python interpreter to perform more complex operations
  
  **Pal: Program-aided language models** *Gao, Luyu, et al.* 2023 [[Paper]](https://dl.acm.org/doi/10.5555/3618408.3618843)
  
  **Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks** *Chen, Wenhu, et al.* 2022.11 [[Paper]](https://openreview.net/forum?id=YfZ4ZPt8zd)
  
  **Mint: Evaluating llms in multi-turn interaction with tools and language feedback** *Wang, Xingyao, et al.* 2023.09 [[Paper]](https://openreview.net/forum?id=jp3gWrMuIZ&referrer=%5Bthe%20profile%20of%20Hao%20Peng%5D(%2Fprofile%3Fid%3D~Hao_Peng4))

  **MATHSENSEI: A Tool-Augmented Large Language Model for Mathematical Reasoning** *Das, Debrup, et al.* 2024 [[Paper]](https://aclanthology.org/2024.naacl-long.54/)

  **ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving** *Gou, Zhibin, et al.* 2023.09 [[Paper]](https://openreview.net/forum?id=Ep0TtjVoap)

- Tools for more advanced business activities, e.g., financial, medical, education, etc.
  
  **On the Tool Manipulation Capability of Open-source Large Language Models** *Xu, Qiantong, et al.* 2023.05 [[Paper]](https://openreview.net/forum?id=iShM3YolRY&referrer=%5Bthe%20profile%20of%20Changran%20Hu%5D(%2Fprofile%3Fid%3D~Changran_Hu1))
  
  **ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases** *Tang, Qiaoyu, et al.* 2023.06 [[Preprint]](https://arxiv.org/abs/2306.05301)
  
  **Mint: Evaluating llms in multi-turn interaction with tools and language feedback** *Wang, Xingyao, et al.* 2023.09 [[Paper]](https://openreview.net/forum?id=jp3gWrMuIZ&referrer=%5Bthe%20profile%20of%20Hao%20Peng%5D(%2Fprofile%3Fid%3D~Hao_Peng4))

  **AgentMD: Empowering Language Agents for Risk Prediction with Large-Scale Clinical Tool Learning** *Jin, Qiao, et al.* 2024.02 [[Paper]](https://arxiv.org/abs/2402.13225)

### Interaction with the world 🌐

- Access real-time or real-world information such as weather, location, etc.
  
  **On the Tool Manipulation Capability of Open-source Large Language Models** *Xu, Qiantong, et al.* 2023.05 [[Paper]](https://openreview.net/forum?id=iShM3YolRY&referrer=%5Bthe%20profile%20of%20Changran%20Hu%5D(%2Fprofile%3Fid%3D~Changran_Hu1))
  
  **ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases** *Tang, Qiaoyu, et al.* 2023.06 [[Preprint]](https://arxiv.org/abs/2306.05301)

- Managing personal events such as calendar or emails
  
  **Toolformer: Language Models Can Teach Themselves to Use Tools** *Schick, Timo, et al.* 2024 [[Paper]](https://openreview.net/forum?id=Yacmpz84TH&referrer=%5Bthe%20profile%20of%20Roberto%20Dessi%5D(%2Fprofile%3Fid%3D~Roberto_Dessi1))

- Tools in embodied environments, e.g., the Minecraft world
  
  **Voyager: An Open-Ended Embodied Agent with Large Language Models** *Wang, Guanzhi, et al.* 2023.05 [[Paper]](https://openreview.net/forum?id=ehfRiF0R3a)

- Tools interacting with the physical world
  
  **ProgPrompt: Generating Situated Robot Task Plans using Large Language Models** *Singh, Ishika, et al.* 2023 [[Paper]](https://openreview.net/forum?id=3K4-U_5cRw)
  
  **Alfred: A benchmark for interpreting grounded instructions for everyday tasks** *Shridhar, Mohit, et al.* 2020 [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shridhar_ALFRED_A_Benchmark_for_Interpreting_Grounded_Instructions_for_Everyday_Tasks_CVPR_2020_paper.pdf)
  
  **Autonomous chemical research with large language models** *Boiko, Daniil A., et al.* 2023 [[Paper]](https://www.nature.com/articles/s41586-023-06792-0)

### Non-textual modalities 🎞️

- Tools providing access to information in non-textual modalities
  
  **Vipergpt: Visual inference via python execution for reasoning** *Surís, Dídac, Sachit Menon, and Carl Vondrick.* 2023 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Suris_ViperGPT_Visual_Inference_via_Python_Execution_for_Reasoning_ICCV_2023_paper.pdf)
  
  **MM-REACT: Prompting ChatGPT for Multimodal Reasoning and Action** *Yang, Zhengyuan, et al.* 2023.03 [[Preprint]](https://arxiv.org/abs/2303.11381)
  
  **AssistGPT: A General Multi-modal Assistant that can Plan, Execute, Inspect, and Learn** *Gao, Difei, et al.* 2023.06 [[Preprint]](https://arxiv.org/abs/2306.08640)

- Tools that can answer questions about data in other modalities
  
  **Visual Programming: Compositional visual reasoning without training** *Gupta, Tanmay, and Aniruddha Kembhavi.* 2023 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.pdf)

### Special-skilled models 🤗

- Text-generation models that can perform specific tasks, e.g., question answering, machine translation
  
  **Toolformer: Language Models Can Teach Themselves to Use Tools** *Schick, Timo, et al.* 2024 [[Paper]](https://openreview.net/forum?id=Yacmpz84TH&referrer=%5Bthe%20profile%20of%20Roberto%20Dessi%5D(%2Fprofile%3Fid%3D~Roberto_Dessi1))
  
  **ART: Automatic multi-step reasoning and tool-use for large language models** *Paranjape, Bhargavi, et al.* 2023.03 [[Preprint]](https://arxiv.org/abs/2303.09014)

- Integration of available models on Huggingface, TorchHub, TensorHub, etc.
  
  **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face** *Shen, Yongliang, et al.* 2024 [[Paper]](https://openreview.net/forum?id=yHdTscY6Ci)
  
  **Gorilla: Large language model connected with massive apis** *Patil, Shishir G., et al.* 2023.05 [[Paper]](https://arxiv.org/abs/2305.15334)
  
  **Taskbench: Benchmarking large language models for task automation** *Shen, Yongliang, et al.* 2023.11 [[Paper]](https://openreview.net/forum?id=70xhiS0AQS&referrer=%5Bthe%20profile%20of%20Xu%20Tan%5D(%2Fprofile%3Fid%3D~Xu_Tan1))

## $\S5$ Advanced methods

### $\S5.1$ Complex tool selection and usage 🧐

- Train retrievers that map natural language instructions to tool documentation
  
  **DocPrompting: Generating Code by Retrieving the Docs** *Zhou, Shuyan, et al.* 2022.07 [[Paper]](https://openreview.net/forum?id=ZTCxT2t2Ru)
  
  **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs** *Qin, Yujia, et al.* 2023.07 [[Paper]](https://openreview.net/forum?id=dHng2O0Jjr)

- Ask LMs to write hypothetical tool descriptions and search relevant tools
  
  **CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets** *Yuan, Lifan, et al.* 2023.09 [[Paper]](https://arxiv.org/abs/2309.17428)

- Complex tool usage, e.g., parallel calls
  
  **Function Calling and Other API Updates** *Eleti, Atty, et al.* 2023.06 [[Blog]](https://openai.com/blog/function-calling-and-other-api-updates)
  
  **An LLM Compiler for Parallel Function Calling** *Kim, Sehoon, et al.* 2023.12 [[Paper]](https://arxiv.org/abs/2312.04511)

### $\S5.2$ Tools in programmatic contexts 👩‍💻

- Domain-specific logical forms to query structured data
  
  **Semantic parsing on freebase from question-answer pairs** *Berant, Jonathan, et al.* 2013 [[Paper]](https://aclanthology.org/D13-1160/)
  
  **Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task** *Yu, Tao, et al.* 2018.09 [[Paper]](https://aclanthology.org/D18-1425/)
  
  **Break It Down: A Question Understanding Benchmark** *Wolfson, Tomer, et al.* 2020 [[Paper]](https://aclanthology.org/2020.tacl-1.13/)

- Domain-specific actions for agentic tasks such as web navigation
  
  **Reinforcement Learning on Web Interfaces using Workflow-Guided Exploration** *Liu, Evan Zheran, et al.* 2018.02 [[Paper]](https://openreview.net/forum?id=ryTp3f-0-)
  
  **WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents** *Yao, Shunyu, et al.* 2022.07 [[Paper]](https://arxiv.org/abs/2207.01206)
  
  **Webarena: A realistic web environment for building autonomous agents** *Zhou, Shuyan, et al.* 2023.07 [[Paper]](https://arxiv.org/abs/2307.13854)

- Using external Python libraries as tools
  
  **ToolCoder: Teach Code Generation Models to use API search tools** *Zhang, Kechi, et al.* 2023.05 [[Paper]](https://arxiv.org/abs/2305.04032)

- Using expert designed functions as tools to answer questions about images
  
  **Visual Programming: Compositional visual reasoning without training** *Gupta, Tanmay, and Aniruddha Kembhavi.* 2023 [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.pdf)
  
  **Vipergpt: Visual inference via python execution for reasoning** *Surís, Dídac, Sachit Menon, and Carl Vondrick.* 2023 [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Suris_ViperGPT_Visual_Inference_via_Python_Execution_for_Reasoning_ICCV_2023_paper.pdf)

- Using GPT as a tool to query external Wikipedia knowledge for table-based question answering
  
  **Binding Language Models in Symbolic Languages** *Cheng, Zhoujun, et al.* 2022.10 [[Paper]](https://openreview.net/forum?id=lH1PV42cbF)

- Incorporate QA API and operation APIs to assist table-based question answering
  
  **API-Assisted Code Generation for Question Answering on Varied Table Structures** *Cao, Yihan, et al.* 2023.12 [[Paper]](https://aclanthology.org/2023.emnlp-main.897)

### $\S5.3$ Tool creation and reuse 👩‍🔬

- Approaches to abstract libraries for domain-specific logical forms from a large corpus
  
  **DreamCoder: growing generalizable, interpretable knowledge with wake--sleep Bayesian program learning** *Ellis, Kevin, et al.* 2020.06 [[Paper]](https://arxiv.org/abs/2006.08381)
  
  **Leveraging Language to Learn Program Abstractions and Search Heuristics]** *Wong, Catherine, et al.* 2021 [[Paper]](https://proceedings.mlr.press/v139/wong21a.html)
  
  **Top-Down Synthesis for Library Learning** *Bowers, Matthew, et al.* 2023 [[Paper]](https://doi.org/10.1145/3571234)
  
  **LILO: Learning Interpretable Libraries by Compressing and Documenting Code** *Grand, Gabriel, et al.* 2023.10 [[Paper]](https://openreview.net/forum?id=TqYbAWKMIe)

- Make and learn skills (Java programs) in the embodied Minecraft world
  
  **Voyager: An Open-Ended Embodied Agent with Large Language Models** *Wang, Guanzhi, et al.* 2023.05 [[Paper]](https://arxiv.org/abs/2305.16291)

- Leverage LMs as tool makers on BigBench tasks
  
  **Large Language Models as Tool Makers** *Cai, Tianle, et al.* 2023.05 [[Preprint]](https://arxiv.org/pdf/2305.17126)

- Create tools for math and table QA tasks by example-wise tool making
  
  **CREATOR: Disentangling Abstract and Concrete Reasonings of Large Language Models through Tool Creation** *Qian, Cheng, et al.* 2023.05 [[Paper]](https://arxiv.org/pdf/2305.14318)

- Make tools via heuristic-based training and tool deduplication
  
  **CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets** *Yuan, Lifan, et al.* 2023.09 [[Paper]](https://arxiv.org/abs/2309.17428)

- Learning tools by refactoring a small amount of programs
  
  **ReGAL: Refactoring Programs to Discover Generalizable Abstractions** *Stengel-Eskin, Elias, Archiki Prasad, and Mohit Bansal.* 2024.01 [[Preprint]](https://arxiv.org/abs/2401.16467)

- A training-free approach to make tools via execution consistency
  
  🎁 **TroVE: Inducing Verifiable and Efficient Toolboxes for Solving Programmatic Tasks** *Wang, Zhiruo, Daniel Fried, and Graham Neubig.* 2024.01 [[Preprint]](https://arxiv.org/abs/2401.12869)

## $\S6$ Evaluation: Testbeds

### $\S6.1.1$ Repurposed existing datasets

- Datasets that require reasoning over texts
  
  **Measuring Mathematical Problem Solving With the MATH Dataset** *Hendrycks, Dan, et al.* 2021.03 [[Paper]](https://arxiv.org/pdf/2103.03874)
  
  **Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models** *Srivastava, Aarohi, et al.* 2022.06 [[Paper]](https://openreview.net/forum?id=uyTL5Bvosj)

- Datasets that require reasoning over structured data, e.g., tables
  
  **Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning** *Lu, Pan, et al.* 2022.09 [[Paper]](https://arxiv.org/pdf/2209.14610)
  
  **Compositional Semantic Parsing on Semi-Structured Tables** *Pasupat, Panupong, and Percy Liang.* 2015 [[Paper]](https://aclanthology.org/P15-1142)
  
  **HiTab: A Hierarchical Table Dataset for Question Answering and Natural Language Generation** *Cheng, Zhoujun, et al.* 2022 [[Paper]](https://aclanthology.org/2022.acl-long.78/)

- Datasets that require reasoning over other modalities, e.g., images and image pairs
  
  **Gqa: A new dataset for real-world visual reasoning and compositional question answering** *Hudson, Drew A., and Christopher D. Manning.* 2019.02 [[Paper]](https://arxiv.org/abs/1902.09506)
  
  **A Corpus for Reasoning about Natural Language Grounded in Photographs** *Suhr, Alane, et al.* 2019 [[Paper]](https://aclanthology.org/P19-1644)

- Example datasets that require retriever model (tool) to solve
  
  **Natural Questions: A Benchmark for Question Answering Research** *Kwiatkowski, Tom, et al.* 2019 [[Paper]](https://aclanthology.org/Q19-1026)
  
  **TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension** *Joshi, Mandar, et al.* 2017 [[Paper]](https://aclanthology.org/P17-1147)

### $\S6.1.2$ Aggregated API benchmarks

- Collect RapidAPIs and use models to synthesize examples for evaluation
  
  **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs** *Qin, Yujia, et al.* 2023.07 [[Paper]](https://openreview.net/forum?id=dHng2O0Jjr)

- Collect APIs from PublicAPIs and use models to synthesize examples
  
  **ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases** *Tang, Qiaoyu, et al.* 2023.06 [[Preprint]](https://arxiv.org/abs/2306.05301)

- Collect APIs from PublicAPIs and manually annotate examples for evaluation
  
  **API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs** *Li, Minghao, et al.* 2023.12 [[Paper]](https://aclanthology.org/2023.emnlp-main.187/)

- Collect APIs from OpenAI plugin list and use models to synthesize examples
  
  **MetaTool Benchmark for Large Language Models: Deciding Whether to Use Tools and Which to Use** *Huang, Yue, et al.* 2023.10 [[Paper]](https://openreview.net/forum?id=R0c2qtalgG&referrer=%5Bthe%20profile%20of%20Neil%20Zhenqiang%20Gong%5D(%2Fprofile%3Fid%3D~Neil_Zhenqiang_Gong1))

- Collect neural model tools from Huggingface hub, TorchHub, and TensorHub
  
  **Gorilla: Large language model connected with massive apis** *Patil, Shishir G., et al.* 2023.05 [[Paper]](https://arxiv.org/abs/2305.15334)

- Collect neural model tools from Huggingface
  
  **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face** *Shen, Yongliang, et al.* 2024 [[Paper]](https://openreview.net/forum?id=yHdTscY6Ci)

- Collect tools from Huggingface and PublicAPIs
  
  **Taskbench: Benchmarking large language models for task automation** *Shen, Yongliang, et al.* 2023.11 [[Paper]](https://openreview.net/forum?id=70xhiS0AQS&referrer=%5Bthe%20profile%20of%20Xu%20Tan%5D(%2Fprofile%3Fid%3D~Xu_Tan1))

- Collect Action Sequences in real-world macOS/iPadOS/iOS.
  
  **ShortcutsBench: A Large-Scale Real-World Benchmark for API-Based Agents** *Shen, Haiyang, et al.* 2024.07 [[Paper]](https://arxiv.org/abs/2407.00132)
