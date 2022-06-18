LaTeX formulas in Markdown files may not be displayed correctly on github, in which case please use the [description in PDF](mathematical_description.pdf).

**Input Data**:

1. $T=\lbrace t_1, ..., t_7 \rbrace$ — Set of trees.

2. $Y(t)=\lbrace y_{t1}, ..., y_{tn_t} \rbrace$ — Set of years for which the measurements for the tree $t$ are available,  $t \in T$

3. $Y=\bigcup_{t \in T} Y(t)$ — Set of all years for which the measurements are available.

4. $T(y)=\lbrace t_{y1}, ..., t_{ym_y} \rbrace$ — Set of trees for which the measurements for the year $y$ are availvable, $y \in Y$

    $\left (T \equiv \bigcup_{y \in Y} T(y) \right )$

5. $e^{raw} = e^{raw}(t,y)=\lbrace e^{raw}_1, ..., e^{raw}_\varepsilon\rbrace$ — Raw tracheid data where:

    $e^{raw}_k = e^{raw}_k(t,y) \in \lbrace d^{raw}_k, c^{raw}_k\rbrace$

    $d^{raw}_k=d^{raw}_k(t,y)$ — Diameter of the $k^{th}$ cell in a raw tracheid

    $c^{raw}_k=c^{raw}_k(t,y)$ — Cell wall thickness of the $k^{th}$ cell in a raw tracheid

    $\varepsilon=\varepsilon(t,y)$ — Number of cells in $e^{raw}(t,y)$

    $k=\overline{1,\varepsilon}$, $t\in T$, $y\in Y(t)$

6. $N$ — Number of cells for tracheid normalization.

**Normalization procedure description:**

For each $e^{raw}$ an intermediate sequence is constructed:

$$e^* = \lbrace\underbrace{e^{raw}_1,...,e^{raw}_1}_{N},\underbrace{e^{raw}_2,...,e^{raw}_2}_{N}, ..., \underbrace{e^{raw}_\varepsilon,...,e^{raw}_\varepsilon}_{N}\rbrace$$


And tracheid data $e = \lbrace e_1, ..., e_N\rbrace$ normalized to $N$ cells are obtained: 

$$e_i = \frac{1}{\varepsilon} \sum_{j=\varepsilon \cdot (i-1)+1}^{\varepsilon \cdot i}e^{*}_j, i=\overline{1, N}$$

Using this procedure the following was obtained:

$d = \lbrace d_1, ..., d_N\rbrace$ — data on the tracheid cell diameters normalized to N cells 

$c = \lbrace c_1, ..., c_N\rbrace$ — data on the tracheid cell wall thicknesses normalized to N cells


**Normalized tracheid description:**

$R(t,y) =d \cup c = \lbrace d_1, ... , d_{N}, c_1, ..., c_{N}\rbrace$ — Tracheid normalized to $N$ cells. Where:

$d_i=d_i(t,y)$ — Diameter of the $i^{th}$ cell in a normalized tracheid

$c_i=c_i(t,y)$ — Cell wall thickness of the $i^{th}$ cell in a normalized tracheids

$i=\overline{1,N}$, $t\in T, y\in Y(t)$


**Description of the methods for forming objects for clustering:**


*Method A*:

1. $$ R^A(y)=\frac{1}{\left| T(y) \right|}\sum_{t\in T(y)}R(t,y), y\in Y $$

2. $$R_{mean}^A=\frac{1}{\sum_{t\in T}\left| Y(t)\right|}\sum_{t\in T}\sum_{y\in Y(t)}R(t,y)$$

3. $$O_A(y)=\frac{R^A(y)}{R_{mean}^A}, y\in Y$$

$O_A(y)$ — object for the year $y$ obtained by *Method A*

*Method B*:

1. $$R^B(t)=\frac{1}{\left| Y(t) \right|}\sum_{y\in Y(t)}R(t,y), t\in T$$

2. $$o_B(t,y)=\frac{R(t,y)}{R^B(t)}, t\in T, y\in Y(t)$$

3. $$O_B(y)=\frac{1}{\left| T(y) \right|}\sum_{t\in T(y)}o_B(t,y), y\in Y$$

$O_B(y)$ — object for the year $y$ obtained by *Method B*