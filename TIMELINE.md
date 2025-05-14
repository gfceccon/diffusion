# Timeline IA Generativa e Diffusion

## Fundamentos e Bases Teóricas (Pré-1990)
- Latent variable model (1984)
  - Fundamento teórico para modelos probabilísticos e geração de dados sintéticos
  - Base para métodos como autoencoders, modelos de mistura e inferência variacional
- Redes neurais multicamadas e algoritmo de retropropagação (backpropagation, 1986)
  - Avanço teórico fundamental para o treinamento de redes profundas
- Modelos probabilísticos e estatísticos
  - Hidden Markov Models (HMMs, 1989)
  - Inferência bayesiana e modelos gráficos aplicados em NLP e reconhecimento de fala

## Consolidação Probabilística e Inverno de IA (1990-2000)
- Inverno de IA
  - Redução de investimentos e avanços lentos em IA
  - Limitações computacionais e de dados restringem o progresso de modelos generativos
- Consolidação de métodos probabilísticos
  - Uso de HMMs, modelos de mistura gaussiana e inferência variacional em aplicações reais
- Artigos e pesquisas fundamentais em aprendizado não supervisionado e modelagem probabilística
  - PCA (Principal Component Analysis): método clássico para redução de dimensionalidade e descoberta de estruturas latentes.
  - K-means e clustering hierárquico: técnicas de agrupamento amplamente usadas para análise exploratória de dados.
  - Mixture of Gaussians (MoG): base para modelos generativos probabilísticos e inspiração para VAEs.
  - EM (Expectation-Maximization): algoritmo central para ajuste de modelos de mistura e inferência em variáveis latentes.
  - ICA (Independent Component Analysis): separação de fontes e descoberta de fatores independentes, influenciando autoencoders e modelos latentes.
  - RBM (Restricted Boltzmann Machines) e Deep Belief Networks: primeiros modelos neurais não supervisionados profundos, precursores de autoencoders e VAEs.

## Deep Learning e Arquiteturas Clássicas (2000-2018)
- Surgimento de técnicas de aprendizado profundo (Deep Learning)
  - LSTM (1997, popularizado nos anos 2000)
    - Permite modelar sequências longas, fundamental para geração de texto e áudio
  - Avanço em autoencoders e redes convolucionais (CNNs)
    - Autoencoders clássicos usados para compressão e reconstrução de dados
    - CNNs revolucionam o processamento de imagens, abrindo caminho para geração visual
- VAE – Variational Autoencoder (2013, 2015)
  - Introdução do truque de reparametrização e geração probabilística de dados
  - Permite interpolação e amostragem em espaços latentes contínuos
- GANs – Generative Adversarial Networks (2014)
  - DCGAN (2015)
  - WGAN (2017)
  - StyleGAN (2018)
- FCN – Fully Convolutional Networks (2014)
  - Aplicações em segmentação e geração de imagens
- LDM – Latent Diffusion Model (2015)
  - Introduz o conceito de geração em espaço latente, reduzindo custo computacional
- U-Net (2015)
  - Arquitetura essencial para tarefas de geração e segmentação de imagens médicas e científicas
- AlignDRAW (2015-2016)
  - Primeiros modelos text-to-image com atenção, combinando geração sequencial e condicionamento textual
  - Marca o início da integração entre linguagem natural e geração visual

## Modelos Generativos Modernos (2018-2022)
- BigGAN (2018-2019)
  - GANs de alta resolução para geração de imagens realistas
- VQ-VAE-2 (2020)
  - Avanço em autoencoders variacionais para geração de imagens de alta qualidade
- DALL-E 1 (2021)
  - Primeira geração de imagens realistas a partir de texto usando transformers
  - Populariza o conceito de modelos multimodais (texto e imagem)
- CLIP (2021)
  - Modelo multimodal que conecta texto e imagem, usado em várias aplicações generativas
- GLIDE (2021)
  - Modelo de diffusion para geração de imagens condicionadas por texto

## Explosão dos Modelos Multimodais e Diffusion (2022-2025)
### Ano 2022
- Imagen (2022)
  - Modelo text-to-image da Google com alta fidelidade, usando grandes modelos de linguagem para codificação textual
- DALL-E 2 (2022)
  - Imagens mais realistas e detalhadas a partir de texto, com maior controle criativo
- Midjourney (2022)
  - IA generativa de imagens com foco artístico e comunidade ativa
- Stable Diffusion (2022)
  - Modelo open source de diffusion, popularizando IA generativa acessível e customizável
- Parti (2022)
  - Modelo text-to-image da Google com foco em geração de imagens detalhadas
- Make-A-Video (2022)
  - Expansão para geração de vídeos realistas a partir de texto
- Imagen Video (2022)
  - Modelo avançado para geração de vídeos com alta fidelidade
- Phenaki (2022)
  - Geração de vídeos longos e consistentes a partir de descrições textuais
- DreamBooth (2022)
  - Personalização de modelos de geração de imagens

### Ano 2023
- ControlNet (2023)
  - Controle refinado sobre geração de imagens com diffusion
- Stable Diffusion XL (2023)
  - Versão avançada do modelo de diffusion, com maior qualidade e controle
- Mistral (2023)
  - Modelos de linguagem e geração com foco em eficiência e qualidade

### Ano 2024
- SD Turbo (2024)
  - Otimização para geração de imagens em tempo real
- Gemini (2024)
  - Modelo multimodal avançado com integração de texto, imagem e vídeo
- Llama 3 (2024)
  - Avanço em modelos de linguagem para suporte a geração multimodal
- Claude 3 (2024)
  - Expansão de capacidades multimodais e geração criativa
- Mistral (2024)
  - Continuação dos avanços em modelos de linguagem e geração

### Ano 2025
- Sora (2025)
  - Nova arquitetura para integração de IA generativa em aplicações do dia a dia
- Modelos multimodais e diffusion avançados (2025)
  - GPT-4, GPT-4o, Llama 3, Mistral, Claude, Gemini
  - Expansão para geração de vídeo, áudio, 3D e integração em aplicações do dia a dia
  - Avanço em personalização, controle de estilo e segurança de geração

## Referências

### Fundamentos e Bases Teóricas
- Kingma, D. P.; Welling, M. (2013). Auto-Encoding Variational Bayes. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- Goodfellow, I. et al. (2014). Generative Adversarial Nets. [NeurIPS](https://papers.nips.cc/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html)
- Ronneberger, O.; Fischer, P.; Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. [MICCAI](https://arxiv.org/abs/1505.04597)
- Radford, A.; Metz, L.; Chintala, S. (2015). Unsupervised Representation Learning with Deep Convolutional GANs. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)
- Arjovsky, M.; Chintala, S.; Bottou, L. (2017). Wasserstein GAN. [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)
- Baghdadlian, Serop The Complete Timeline of Text-to-Image Evolution!. [Imagem](https://ai.plainenglish.io/the-complete-timeline-of-text-to-image-evolution-b63298234ed6) Acessado 14/05/2025

### Modelos Generativos Modernos
- Karras, T. et al. (2018). A Style-Based Generator Architecture for Generative Adversarial Networks. [CVPR](https://arxiv.org/abs/1812.04948)
- Brock, A.; Donahue, J.; Simonyan, K. (2018). Large Scale GAN Training for High Fidelity Natural Image Synthesis. [ICLR (BigGAN)](https://arxiv.org/abs/1809.11096)
- Razavi, A.; van den Oord, A.; Vinyals, O. (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2. [NeurIPS](https://arxiv.org/abs/1906.00446)
- Ramesh, A. et al. (2021). Zero-Shot Text-to-Image Generation. [arXiv:2102.12092](https://arxiv.org/abs/2102.12092) (DALL-E 1)
- Ramesh, A. et al. (2022). Hierarchical Text-Conditional Image Generation with CLIP Latents. [arXiv:2204.06125](https://arxiv.org/abs/2204.06125) (DALL-E 2)
- Nichol, A.; Dhariwal, P. (2021). GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models. [arXiv:2112.10741](https://arxiv.org/abs/2112.10741)
- Saharia, C. et al. (2022). Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. [arXiv:2205.11487](https://arxiv.org/abs/2205.11487) (Imagen)
- Rombach, R. et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. [CVPR (Stable Diffusion)](https://arxiv.org/abs/2112.10752)
- Esser, P. et al. (2021). Taming Transformers for High-Resolution Image Synthesis. [CVPR (VQGAN)](https://arxiv.org/abs/2012.09841)
- Chen, M. et al. (2021). Pre-Training of Image Transformers for Text-to-Image Generation. [ICML (Parti)](https://arxiv.org/abs/2206.10789)

### Modelos Multimodais e Diffusion Recentes
- Balaji, Y. et al. (2022). Make-A-Video: Text-to-Video Generation without Text-Video Data. [arXiv:2209.14792](https://arxiv.org/abs/2209.14792)
- Singer, A. et al. (2022). Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors. [arXiv:2203.13131](https://arxiv.org/abs/2203.13131)
- Google Research. (2022). Imagen Video: High Definition Video Generation with Diffusion Models. [Site](https://imagen.research.google/video/)
- Google Research. (2022). Phenaki: Variable Length Video Generation. [Site](https://phenaki.video/)
- Ruiz, N. et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. [arXiv:2208.12242](https://arxiv.org/abs/2208.12242)
- Zhang, X. et al. (2023). Adding Conditional Control to Text-to-Image Diffusion Models. [arXiv:2302.05543](https://arxiv.org/abs/2302.05543) (ControlNet)
