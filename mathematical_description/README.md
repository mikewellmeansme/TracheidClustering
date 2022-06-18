$\LaTeX$ formulas in Markdown files may not be displayed correctly on github, in which case please use the [description in PDF](mathematical_description/mathematical_description.pdf).

**Input Data**:

1. $T=\lbrace t_1, \dots, t_7 \rbrace$ — Set of trees.

2. $Y(t)=\lbrace y_{t1}, \dots, y_{tn_t} \rbrace$ — Set of years for which the measurements for the tree $t$ are available,  $t \in T$.

3. $Y=\bigcup_{t \in T} Y(t)$ — Set of all years for which the measurements are available.

4. $T(y)=\lbrace t_{y1}, \dots, t_{ym_y} \rbrace$ — Set of trees for which the measurements for the year $y$ are availvable, $y \in Y$.

    $\left (T \equiv \bigcup_{y \in Y} T(y) \right )$

5. $e^{raw} = e^{raw}(t,y)=\lbrace e^{raw}_1, \dots, e^{raw}_\varepsilon\rbrace$ — Raw tracheid data where:

    $e^{raw}_k = e^{raw}_k(t,y) \in \lbrace d^{raw}_k, c^{raw}_k\rbrace$

    $d^{raw}_k=d^{raw}_k(t,y)$ — Diameter of the $k^{th}$ cell in a raw tracheid

    $c^{raw}_k=c^{raw}_k(t,y)$ — Cell wall thickness of the $k^{th}$ cell in a raw tracheid

    $\varepsilon=\varepsilon(t,y)$ — Number of cells in $e^{raw}(t,y)$

    $k=\overline{1,\varepsilon}$, $t\in T$, $y\in Y(t)$.

6. $N$ — Number of cells for tracheid normalization.

**Normalization procedure description:**

For each $e^{raw}$ an intermediate sequence is constructed:

$$e^* = \lbrace\underbrace{e^{raw}_1,\dots,e^{raw}_1}_{N},\underbrace{e^{raw}_2,\dots,e^{raw}_2}_{N}, \dots, \underbrace{e^{raw}_\varepsilon,\dots,e^{raw}_\varepsilon}_{N}\rbrace$$


And tracheid data $e = \lbrace e_1, \dots, e_N\rbrace$ normalized to $N$ cells are obtained: 

$$e_i = \frac{1}{\varepsilon} \sum_{j=\varepsilon \cdot (i-1)+1}^{\varepsilon \cdot i}e^{*}_j, i=\overline{1, N}$$

Using this procedure the following was obtained:

$d = \lbrace d_1, \dots, d_N\rbrace$ — data on the tracheid cell diameters normalized to N cells 

$c = \lbrace c_1, \dots, c_N\rbrace$ — data on the tracheid cell wall thicknesses normalized to N cells


**Normalized tracheid description:**

$R(t,y) =d \cup c = \lbrace d_1, \dots , d_{N}, c_1, \dots, c_{N}\rbrace$ — Tracheid normalized to $N$ cells. Where:

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

**Description of the Area index**

**Input Data:**

1. $Y_{clim}=\{y_1, \dots, y_{|Y|}\}$  — Set of years for which the climatic measurements are available.

2. $\mathbb{T}(y) = \{T_1, \dots, T_{366}\}$ — Set of daily temperatute for year $y$, where $T_i = T_i(y)$ — temperature in $i^{th}$ day of the year $y$,.

2. $\mathbb{P}(y) = \{P_1, \dots, P_{366}\}$ — Set of daily precipitation for year $y$, where $P_i = P_i(y)$ — precipitation in $i^{th}$ day of the year $y$.

    $i \in \overline{1, 366}$, $y \in Y_{clim}$.
    
    If the data for the day are absent it replaced with the $0$ for the cumulative sum operation and ignored in other cases.

**Preparation description:**

1. $\mathbb{P}^{C}(y) = \{P^{C}_1, \dots, P^{C}_{366}\}$ — Set of cumulative sums of presipitation for year $y$, where:

    $$P^{C}_i = P^{C}_i(y) = \sum_{k=1}^{i}P_i(y)$$
    $i \in \overline{1, 366}$, $y \in Y_{clim}$.

2. $\mathbb{T}^R(y) = \{T^R_1, \dots, T^R_{366}\}$,
    $\mathbb{P}^R(y) = \{P^R_1, \dots, P^R_{366}\}$ — Sets of temperature and cumulative precipitation the year $y$, smoothed with 7-day rolling mean (moving average), where:
    
    $$T^R_i=T^R_i(y)=\frac{1}{7}\sum_{k=i-3}^{i+3}T_k(y)$$
    $$P^R_i=P^R_i(y)=\frac{1}{7}\sum_{k=i-3}^{i+3}P^C_k(y)$$

    $i \in \overline{1, 366}$, $y \in Y_{clim}$.
    
    If $k$ is less than $1$: $T_k(y)=T_{366-k}(y-1)$, $P^C_k(y)=P^C_{366-k}(y-1)$
    
    If $k$ is greater than $366$: $T_k(y)=T_{k-366}(y+1)$, $P^C_k(y)=P^C_{k-366}(y+1)$


3. $\mathbb{T}^S(y) = \{T^S_{\alpha}, \dots, T^S_{\omega}\}$,
 $\mathbb{P}^S(y) = \{P^S_{\alpha}, \dots, P^S_{\omega}\}$ — Sets of temperature and precipitation for the year $y$ scaled with MinMax approach:

    $$T_{min} = \min_{y \in Y_{clim}}\min_{i \in \overline{\alpha, \omega}}\{T_i^R(y)\}, T_{max} = \max_{y \in Y_{clim}}\max_{i \in \overline{\alpha, \omega}}\{T_i^R(y)\}$$

    $$T_{i}^{S}=T_{i}^{S}(y)=\frac{T_i^R(y)-T_{min}}{T_{max}-T_{min}}$$

    $$P_{min} = \min_{y \in Y_{clim}}\min_{i \in \overline{\alpha, \omega}}\{P_i^R(y)\}, P_{max} = \max_{y \in Y_{clim}}\max_{i \in \overline{\alpha, \omega}}\{P_i^R(y)\}$$

    $$P_{i}^{S}=P_{i}^{S}(y)=\frac{P_i^R(y)-P_{min}}{P_{max}-P_{min}}$$

    $\alpha$ — First day of the growth season
    
    $\omega$ — Last day of the growth season

    $i \in \overline{\alpha, \omega}$, $y \in Y_{clim}$.

**Area formula:**

$$Area(y) = \sum_{i=\alpha}^{\omega}|T_{i}^{S}(y)-P_{i}^{S}(y)|$$