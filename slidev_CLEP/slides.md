---
# try also 'default' to start simple
theme: seriph
colorSchema: light
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
#background: https://cover.sli.dev
background:
# some information about your slides, markdown enabled
title: Contrastive Language-Entity Pre-training for Richer Knowledge Graph Embedding
info: |
  ## CLEP paper
# apply any unocss classes to the current slide
author: Andrea Papaluca
class: text-center
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# https://sli.dev/guide/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/guide/syntax#mdc-syntax
mdc: true
---

## Contrastive Language-Entity Pre-training for Richer Knowledge Graph Embedding

**Andrea Papaluca**, Daniel Krefl, Artem Lensky, Hanna Suominen 

<img src="https://download.logo.wine/logo/Australian_National_University/Australian_National_University-Logo.wine.png" style="position:relative; left:275px; top:40px" width="300" height="300" />

<div class="abs-br m-6 flex gap-2">
  <a href="https://github.com/BrunoLiegiBastonLiegi/CLEP" target="_blank" alt="GitHub" title="Open in GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---
transition: fade-out
---

# The CLEP Architecture

$$
(Barack\;Obama,\, born\_in,\, Honolulu)
$$

<img id="clepImage" src="/CLEP_pretraining.svg" style="position:relative; left:150px; top:1px" width="600" height="600" />
  

---
transition: slide-up
---

# A Forward Pass


<div class="grid grid-cols-2 gap-4">
<div>

 $e^{head}$ :  head node of the relational triplet 

 $d^{tail}$ :  description of the tail node 


<br>
<div v-click>
$$
(h_i^{(g)}, \rho_i^{(g)}) = \text{GraphEncoder}\big(e^{head}_i,r_i\big)
$$
</div>
<br>
<div v-click>
$$
x_i^{(g)} = h_i^{(g)} + \rho_i^{(g)}
$$
</div>
<br>
<div v-click>
$$
\tilde{x}_i^{(g)} = \text{MLP}_g\big(x_i^{(g)}\big)
$$
</div>
	
</div>

<div>

Batch of KG triplets 
$$
\big\{\big(e^{head}_{1},r_1,d^{tail}_{1}\big), \, \ldots \, , \big(e^{head}_n,r_n,d^{tail}_n\big)\big\}\;
$$


<br>
<br>
<br>
<br>


<div v-click='1'>
<img src="/graph_encoder.svg" style="position:absolute; left:500px; top:250px" width="300" height="300" />
</div>
<div v-click='2'>
<img src="/head_rel_composition.svg" style="position:absolute; left:810px; top:320px" width="75" height="75" />
</div>
<div v-click='3'>
<img src="/mlp.svg" style="position:absolute; left:895px; top:320px" width="20" height="20" />
</div>

</div>

</div>


---

# A Forward Pass

<div class="grid grid-cols-2 gap-4">

<div>
$$
x_i^{(t)} = \text{TextEncoder}\big(d^{tail}_i\big)
$$
<div v-click='1'>
$$
\tilde{x}_i^{(t)} = \text{MLP}_t\big(x_i^{(t)}\big)
$$
</div>
<div v-click='2'>
<img src="/matrix.svg" style="position:absolute; left:100px; top:275px" width="300" height="300" />
</div>
</div>

<div>

<div v-click='0'>
<img src="/text_encoder.svg" style="position:absolute; left:500px; top:25px" width="300" height="300" />
</div>
<div v-click='1'>
<img src="/mlp.svg" style="position:absolute; left:810px; top:133px" width="20" height="20" />
</div>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<div v-click='2'>
Cosine similarity matrix
$$
m_{i,j} = \frac{\tilde{x}_i^{(g)}\cdot \tilde{x}_j^{(t)}}{\|\tilde{x}_i^{(g)}\|\|\tilde{x}_j^{(t)}\|}\cdot e^\tau
$$
</div>
<br>
<div v-click='3'>

 $\tau$ : temperature scaling the logits

</div>


</div>

</div>

---
transition: fade-out
---

# A Forward Pass
<br>

Row-wise Cross Entropy (CE)

$$
\text{CE}\big(M\big) = -\frac{1}{n} \sum_{i=1}^n \log\frac{e^{m_{i,i}}}{\sum_{j=1}^n e^{m_{i,j}}}
$$

<div v-click.hide>
<img src="/row-wise_CE.svg" style="position:absolute; left:710px; top:133px" width="150" height="150" />
</div>

<br>
<div v-click='1'>

 The column-wise CE is obtained by simply taking $\;M\rightarrow M^T$
 
 <img src="/col-wise_CE.svg" style="position:absolute; left:710px; top:133px" width="150" height="150" />


</div>
<br>
<div v-click='2'>

$$
\mathcal{L}=\frac{1}{2}\bigg(\text{CE}\big(M\big)+\text{CE}\big(M^\top\big)\bigg)
$$

$\longrightarrow$ Enforces minimization of incorrect entity-description associations simultaneously in rows and columns!

</div>

---

# The aligned Text-Graph space
<br>
Euclidean distance of the correct/incorrect entity-description associations

$$
\textcolor{green}{P\bigg(\|\tilde{x}_i^{(g)}-\tilde{x}_i^{(t)}\|\bigg)} \quad\quad \textcolor{red}{P\bigg(\|\tilde{x}_i^{(g)}-\tilde{x}_j^{(t)}\|_{i\neq j}\bigg)}
$$


<div class="grid grid-cols-2 gap-4">

<div v-click='1'>
<img src="/euclidean_dist_FB15k-237-cut_untrained.svg" style="position:absolute; left:50px; top:250px" width="350" height="350" />
</div>

<div>
<div v-click='2'>
<v-drag-arrow pos="435,360,86,0" right op70 />

<v-drag pos="440,320,86,0" text-xl>
  CLEP
</v-drag>
 
<img src="/euclidean_dist_FB15k-237-cut.svg" style="position:absolute; left:550px; top:250px" width="350" height="350" />
</div>
</div>

</div>

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<div v-click='1'>
<div align="center">FB15k-237</div>
</div>

<div v-click='3'>
<v-drag-arrow pos="720,323,77,0" two-way op70 />
<v-drag pos="716,284,120,40" text>
  2x farther
</v-drag>

</div>

---
transition: fade-out
---

# The aligned Text-Graph space

<div class="grid grid-cols-2 gap-4">

<div>
<br>
<div align="center">YAGO3-10</div>
<img src="/euclidean_dist_YAGO3-10-cut.svg" style="position:absolute; left:70px; top:150px" width="350" height="350" />
<div v-click='1'>
<img src="/euclidean_dist_YAGO3-10-cut_overlap.svg" style="position:absolute; left:70px; top:150px" width="350" height="350" />
<v-drag-arrow pos="309,323,216,-146" right op70 color="red"/>
</div>
</div>

<div>

<div v-click='1' align="center">

Incorrect pairs closer than correct ones

 $$\|\tilde{x}_i^{(g)}-\tilde{x}_i^{(t)}\|\ \geq \;\|\tilde{x}_i^{(g)}-\tilde{x}_j^{(t)}\|_{i\neq j} $$
 
</div>
<br>
<br>
<div v-click='2' align="center">

Many descriptions are shared over different entities

```mermaid

flowchart TD;

A("Horror Music\n(Q54864794)") --> C>"music genre."]
B("Party Music\n(Q20022908)") --> C>"music genre."]
```
</div>


</div>


</div>

---
transition: slide-up
---

# Link Prediction across spaces

<div class="grid grid-cols-[550px_400px] gap-4">

<div>

<div>

```mermaid

graph LR;

A(("Barack\n Obama")) 
B(("Michelle\nObama"))
C((Honolulu))
D((Hawaii))
E((Chicago))
F((Illinois))
G((USA))
A --spouse--> B
B --spouse--> A
A --birthplace--> C
B --birthplace--> E
C --located_in--> D
E --located_in--> F
D --located_in--> G
F --located_in--> G
B --nationality--> G
A --president_of--> G
A .- ? .-> G
```

</div>

<br>

<div v-click='1' align="center">
What's Barack Obama's Nationality?

$$
f_s(\text{Barack Obama},\;\text{nationality},\; v) \quad \forall v\in\mathcal{G}
$$

</div>

</div>

<div>

<div v-click='2' style="font-size:60%;">

| <div style="width:15px">Rank</div> | $f_s$ | Link |
| :--- | :--- | :--- |
| 1 | 0.91 | (Barack Obama, nationality, USA) |
| 2 | 0.53 | (Barack Obama, nationality, Hawaii) | 
| 3 | 0.44 | (Barack Obama, nationality, Illinois) | 
| . | . | . |
| . | . | . |
| . | . | . |
| n | 0.11 | (Barack Obama, nationality, Michelle Obama) | 

</div>

</div>

<div>



</div>

</div>

---

# Link Prediction across spaces

- CLEP is trained to align head entities with tails descriptions $\quad e^{head} + r \sim d^{tail}$

$$
f_s(\text{Barack Obama},\;\text{nationality},\; d(v)) \quad \forall v\in\mathcal{G}
$$


<div v-click="1">

<v-drag pos="10,243,120,40" text>

  $$
  \text{node} \in \text{graph space}
  $$
   
</v-drag>

<v-drag pos="483,233,120,40" text>

  $$
  \text{description} \in \text{text space}
  $$
   
</v-drag>

```mermaid
graph LR;

A(("Barack  \n  Obama"))
B>Country primarily located in North America.]
A -. nationality? .-> B
```

</div>

<div class="grid grid-cols-2 gap-4">

<div v-click='2' align="left">
<br>
<br>
Cosine Similarity score


$f_s(h,r,t) = \;\frac{\text{MLP}_g\big(x_{head}^{(g)}\big)\;\cdot\;\text{MLP}_t\big(x_{tail}^{(t)}\big)}{\|\text{MLP}_g\big(x_{head}^{(g)}\big)\|\;\|\text{MLP}_t\big(x_{tail}^{(t)}\big)\|}$

</div>

<div v-click='3'>
<br>

|  | MR | MRR | hits@1 | hits@10 |
| - | - | --- | ------ | ------- |
| CompGCN<sub>CLEP</sub> | **198** | 0.222 | 0.137 | 0.396 |
| RGCN + Distmult | 315 | **0.237** | **0.156** | **0.407** |


<v-drag pos="821,465,120,40" text>
FB15k-237
</v-drag>

</div>

</div>

---

# Link Prediction Finetuning

- Pretrain with CLEP

<div v-click="[0,2]">
<img id="clepImage" src="/CLEP_pretraining.svg" style="position:absolute; left:212px; top:123px" width="600" height="600" />
</div>

<div v-click="[1,2]">
<img src="/te_to_ge_arrow.svg" style="position:absolute; left:465px; top:290px" width="105" height="105" />
</div>

<div v-click='2'>
<img src="/lp_finetuning.svg" style="position:absolute; left:218px; top:149px" width="335" height="335" />
</div>
<div v-click='3'>
<img src="/lp_head.svg" style="position:absolute; left:565px; top:210px" width="43" height="43" />
</div>


<div v-click='3'>
<v-drag-arrow pos="622,250,53,1" right op70 />


<v-drag pos="688,216,320,40" text>

$f_s$

</v-drag>

<v-drag pos="225,427,320,40" text>

$$ f_s(h,r,t)=h^TM_rt \quad\quad\quad\quad\quad\quad f_s(h,r,t)=\| h+r-t \|$$

</v-drag>

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>


- Finetune on pure LP

<v-drag pos="243,397,320,40" text>
RESCAL / DistMult
</v-drag>

<v-drag pos="596,395,320,40" text>
TransE
</v-drag>


</div>

---
transition: fade-out
---

# Link Prediction Finetuning

<v-drag pos="700,4,320,40" text>

<p> &#128311 Randomly initialized model </p>
<p> &#128310 CLEP pretrained model </p>

</v-drag>


<div v-click='1'>
FB15k-237
<img src="/lp_metrics_fb15k-237_compgcn.svg" style="position:absolute; left:60px; top:120px" width="800" height="800" />
</div>
<div v-click='1'>
<v-drag pos="857,175,120,40" text>

$$\sim +1\,\text{-}\,2 \%$$

</v-drag>
</div>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<div v-click='2'>
YAGO3-10
<img src="/lp_metrics_yago3-10_compgcn.svg" style="position:absolute; left:60px; top:330px" width="800" height="800" />
</div>
<div v-click='2'>
<v-drag pos="858,381,120,40" text>

$$\sim +4\,\text{-}\,10 \%$$

</v-drag>
</div>

---

# Conclusion and Future Work

<v-clicks every="1">

- CLEP allows for learning an aligned multi-modal Text-Graph space
- Nodes and corresponding textual descriptions are embedded close in this space 
- Properties of the original spaces are preserved: *e.g.* composition of entities and relations  
- Some of the textual information is transferred to the graph encoder during the pre-training  
&rarr; improved performance on downstream tasks without additional textual inputs
- Is the text encoder expected to manifest a similar transfer?  
&rarr; finetuning for instance on Question Answering?
- Any zero-shot capability enabled?  
&rarr; zero-shot Entity Linking: comparison of entity mentions and node embeddings
- Stable diffusion based Graph Generative Model for Information Extraction

</v-clicks>

--- 

<br>
<br>
<br>
<br>
<div class="center">
  <div>
    <h1>Thank your for the Attention!</h1>
  </div>
</div>

