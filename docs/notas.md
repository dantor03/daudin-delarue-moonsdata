El entrenamiento en detalle

  1. La función de coste

  $$J(\theta) = \underbrace{\mathcal{L}{\text{BCE}}(\theta)}{\text{término 1}} + \varepsilon
  \cdot \underbrace{\mathcal{E}(\theta)}_{\text{término 2}}$$
Término 1 — BCE                                                                             
                                                                                        
$$\mathcal{L}{\text{BCE}} = -\frac{1}{N}\sum{i=1}^{N} \left[ y_i \log \sigma(\text{logit}_i)
   + (1-y_i)\log(1-\sigma(\text{logit}_i)) \right]$$                                        
                                                                                              
  donde $\text{logit}_i = W \cdot X_T^i + b$ es la salida del clasificador lineal sobre el    
  punto $i$ después de integrar la ODE, e $y_i \in {0, 1}$ es su etiqueta.                    
                                                                                              
  En cada término de la suma, los dos sumandos se anulan mutuamente según la etiqueta:        
   
  - Si $y_i = 1$: solo queda $\log \sigma(\text{logit}_i)$. Si el modelo predice bien,        
  $\sigma(\text{logit}_i) \approx 1$ y $\log(1) = 0$. Si falla, $\sigma(\text{logit}_i)
  \approx 0$ y $\log(0) \to -\infty$.                                                         
  - Si $y_i = 0$: solo queda $\log(1 - \sigma(\text{logit}_i))$. Análogamente.
                                                                                              
  El signo negativo delante convierte el $-\infty$ en $+\infty$, de modo que minimizar        
  $\mathcal{L}_{\text{BCE}}$ equivale a maximizar la log-verosimilitud del modelo. En otras   
  palabras, es la media del coste de clasificación sobre los $N$ puntos. 
  ---
  Término 2 — Penalización entrópica

  $$\mathcal{E}(\theta) = \frac{1}{N_p}\sum_j \left[0.05,\theta_j^4 + 0.5,\theta_j^2\right]$$

  donde la suma recorre todos los $N_p$ parámetros de MeanFieldVelocity (W1 y W0), y el
  $1/N_p$ normaliza para que el valor no dependa del tamaño del modelo.

  Este término aproxima la divergencia KL entre la distribución actual de parámetros y el
  prior $\nu^\infty \propto e^{-\ell(a)}$ con $\ell(a) = 0.05|a|^4 + 0.5|a|^2$. La KL completa
   sería:

  $$\text{KL}(\nu_t | \nu^\infty) = \mathbb{E}_{\nu_t}[\ell(a)] - H(\nu_t)$$

  Como los parámetros son estimadores puntuales (no distribuciones), la entropía $H(\nu_t)$ no
   es computable — la distribución es una suma de deltas de Dirac con entropía $-\infty$. Solo
   se puede calcular el término de energía $\mathbb{E}_{\nu_t}[\ell(a)]$, que es exactamente
  $\mathcal{E}(\theta)$.

  El término $0.5,\theta_j^2$ es regularización L2 estándar (weight decay). El término
  $0.05,\theta_j^4$ es lo que hace que el prior sea supercoercivo: crece más rápido que
  cuadrático, lo que garantiza la desigualdad log-Sobolev para $\nu^\infty$, que a su vez
  implica la condición PL. Sin el término cuártico no hay garantías teóricas.

  El coeficiente $\varepsilon$ controla el balance:
  - $\varepsilon = 0$: solo BCE, sin garantías
  - $\varepsilon$ pequeño: garantías teóricas con poco sacrificio en clasificación
  - $\varepsilon$ grande: parámetros muy concentrados cerca del origen, fronteras más suaves,
  $J^*$ más alto

  ---
  2. Adam

  Adam mantiene para cada parámetro $\theta_j$ dos estimadores:

  $$m_j \leftarrow \beta_1 m_j + (1-\beta_1)\nabla_j J \quad \text{(media del gradiente)}$$
  $$v_j \leftarrow \beta_2 v_j + (1-\beta_2)(\nabla_j J)^2 \quad \text{(varianza del
  gradiente)}$$

  Y el paso es:

  $$\theta_j \leftarrow \theta_j - \eta_t \cdot \frac{\hat{m}_j}{\sqrt{\hat{v}_j} +
  \epsilon}$$

  El efecto es que cada parámetro tiene su propio learning rate adaptativo: parámetros con
  gradiente grande dan pasos pequeños, parámetros con gradiente pequeño dan pasos grandes. Con
   lr=0.005, $\beta_1=0.9$, $\beta_2=0.999$.

  ---
  3. Cosine annealing

  El learning rate se reduce desde lr_max=0.005 hasta ~0 siguiendo un coseno:

  $$\eta_t = \frac{\eta_{\max}}{2}\left(1 + \cos\left(\frac{\pi \cdot
  t}{T_{\max}}\right)\right)$$

  época 0    → lr = 0.005
  época 400  → lr ≈ 0.0025
  época 800  → lr ≈ 0.0

  Las curvas de pérdida se aplanan al final no porque la condición PL deje de cumplirse, sino
  porque el lr se ha reducido. Por eso grad_norm2 se mide con el gradiente real,
  independientemente del lr del scheduler.

  ---
  4. Gradient clipping

  Si la norma global de todos los gradientes supera max_norm=5.0:

  $$\text{si } |\nabla J| > 5 \quad \Rightarrow \quad \nabla J \leftarrow \nabla J \cdot
  \frac{5}{|\nabla J|}$$

  No cambia la dirección, solo la magnitud. Necesario en las primeras épocas cuando los
  parámetros están lejos del mínimo y los gradientes pueden ser grandes.
Qué es grad_norm2                                                                           

  Es la norma al cuadrado del gradiente global:                                               
   
  $$|\nabla J|^2 = \sum_j \left(\frac{\partial J}{\partial \theta_j}\right)^2$$               
                  
  donde la suma recorre todos los parámetros escalares del modelo — W1, W0 y el clasificador. 
  Es simplemente la suma de todos los gradientes individuales al cuadrado.
                                                                                              
  En el código:   

  gn2 = sum(p.grad.pow(2).sum().item()
            for p in model.parameters() if p.grad is not None)                                
   
  Para cada tensor de parámetros, eleva cada gradiente escalar al cuadrado y los suma todos.  
  El resultado es un único número.
                                                                                              
  Se registra en hist['grad_norm2'] y se usa en el experimento C para verificar la condición  
  PL:
                                                                                              
  $$|\nabla J|^2 \geq 2\mu (J - J^*)$$                                                        
   
  Si en cada época el ratio $|\nabla J|^2 / (2(J - J^*))$ se mantiene positivo, la condición  
  se cumple.      
                                                                                              
  ---             
  Gradient clipping en detalle
                              
  Después de medir grad_norm2, se calcula la norma (no al cuadrado):
                                                                                              
  $$|\nabla J| = \sqrt{\sum_j \left(\frac{\partial J}{\partial \theta_j}\right)^2}$$          
                                                                                              
  Si esta norma supera max_norm=5.0, todos los gradientes se reescalan proporcionalmente:     
                  
  $$\frac{\partial J}{\partial \theta_j} \leftarrow \frac{\partial J}{\partial \theta_j} \cdot
   \frac{5}{|\nabla J|}$$
                                                                                              
  El factor es el mismo para todos los parámetros, así que la dirección del gradiente no      
  cambia — solo la magnitud. Es como normalizar el vector gradiente y multiplicarlo por 5.
                                                                                              
  ---             
  Por qué es necesario aquí concretamente
                                                                                              
  El gradiente llega a W1 y W0 sumando contribuciones de las 40 evaluaciones de $F$ (10 pasos
  RK4 × 4 evaluaciones por paso). En las primeras épocas, cuando los parámetros están lejos   
  del mínimo, esas 40 contribuciones pueden acumularse y producir un gradiente muy grande.
                                                                                              
  Si Adam aplicara ese gradiente sin clipar, el paso sería enorme y los pesos cambiarían tanto
   que en el siguiente forward pass la ODE explotaría numéricamente — RK4 integraría un campo
  con pesos descontrolados y $X_T$ tendría valores del orden de $10^6$.                       
                  
  Con max_norm=5.0 se garantiza que por muy grande que sea el gradiente, el paso efectivo que 
  da Adam está acotado.
                                                                                              
  ---             
  La interacción con Adam
                                                                                              
  Adam no aplica el gradiente directamente — lo escala por $\hat{m}_j / \sqrt{\hat{v}_j}$.
  Pero si el gradiente raw es muy grande, $\hat{m}_j$ y $\hat{v}_j$ también lo son, y aunque  
  Adam lo amortigua parcialmente no lo controla completamente en las primeras épocas cuando
  los estimadores de media y varianza aún no están bien calibrados. El clipping actúa antes de
   que Adam actualice esos estimadores, así que los protege también de acumular valores
  extremos en su memoria interna
  ---
  5. El orden exacto en cada época

  opt.zero_grad()                          # 1. borrar gradientes anteriores
  loss, lt, lr_val = model.compute_loss()  # 2. forward: ODE + BCE + penalización
  loss.backward()                          # 3. backward: ∂J/∂θ para todos los params

  gn2 = sum(p.grad.pow(2).sum() ...)       # 4. medir ‖∇J‖² ANTES de modificar
                                           #    (numerador de la condición PL)

  clip_grad_norm_(model.parameters(), 5.0) # 5. clipping
  opt.step()                               # 6. Adam actualiza θ
  sch.step()                               # 7. cosine annealing actualiza lr

  El paso 4 va antes del clipping porque grad_norm2 debe reflejar el gradiente teórico real —
  si se midiera después, los valores estarían artificialmente reducidos y la estimación de
  $\mu$ estaría sesgada a la baja.



--------------------------
Por qué $H(\nu_t)$ no es computable
                                                                                        
  La entropía diferencial                      

  Para una medida de probabilidad continua $\nu_t$ con densidad $p(a)$, la entropía           
  diferencial es:
                                                                                              
  $$H(\nu_t) = -\int_A p(a) \log p(a) , da$$                                                  
   
  Esta cantidad está bien definida cuando $\nu_t$ tiene densidad respecto a la medida de      
  Lebesgue — es decir, cuando es una distribución "suave" como una Gaussiana.
                                                                                              
  ---                                                                                         
  El problema: $\nu_t$ es una suma de deltas
                                                                                              
  En la implementación, los M parámetros son valores fijos $a^1, \ldots, a^M \in A$. La
  distribución empírica es:                                                                   
   
  $$\nu_t = \frac{1}{M}\sum_{m=1}^M \delta_{a^m}$$                                            
                                                            
  Una delta de Dirac $\delta_{a^m}$ no tiene densidad respecto a Lebesgue — es una medida     
  singular, concentrada en un único punto. Formalmente su "densidad" sería infinita en $a^m$ y
   cero en todo lo demás.                                                                     
                                                            
  ---
  Qué le pasa a $H$
                                                                                              
  Para calcular $H(\nu_t)$ necesitarías:
                                                                                              
  $$H(\nu_t) = -\int_A p(a)\log p(a), da$$                                                    
                                                                                              
  Pero $p(a)$ no existe — no hay función que describa la densidad de $\frac{1}{M}\sum_m       
  \delta_{a^m}$ respecto a Lebesgue. Alternativamente puedes usar la definición discreta:
                                                                                              
  $$H(\nu_t) = -\sum_{m=1}^M \frac{1}{M} \log \frac{1}{M} = \log M$$                          
   
  Pero eso es la entropía de la distribución sobre los índices $m$, no sobre el espacio $A$.  
  No captura cuánto se dispersan los parámetros en $A$ — dos modelos con parámetros muy
  distintos tendrían el mismo $H = \log M$.                                                   
                                                            
  ---
  La KL completa
                                                                                              
  La KL entre $\nu_t$ y el prior continuo $\nu^\infty$ es:
                                                                                              
  $$\text{KL}(\nu_t | \nu^\infty) = \int_A \log\frac{d\nu_t}{d\nu^\infty}(a), d\nu_t(a)$$     
                                                                                              
  Para que esto esté definido, $\nu_t$ debe ser absolutamente continua respecto a $\nu^\infty$
   — es decir, $\nu_t$ no puede tener masa en lugares donde $\nu^\infty$ tiene masa cero. Pero
   $\nu^\infty$ es continua y $\nu_t$ es discreta: tiene toda su masa concentrada en M puntos.
   La derivada de Radon-Nikodym $d\nu_t/d\nu^\infty$ no existe, y la KL es técnicamente
  $+\infty$.

  ---
  Por qué sí se puede calcular $\mathbb{E}_{\nu_t}[\ell(a)]$
                                                                                              
  Aunque la KL completa sea $+\infty$, el término de energía sí es computable porque es
  simplemente una media sobre los M puntos:                                                   
                                                            
  $$\mathbb{E}{\nu_t}[\ell(a)] = \int_A \ell(a), d\nu_t(a) = \frac{1}{M}\sum{m=1}^M \ell(a^m) 
  = \frac{1}{M}\sum_{m=1}^M \left[0.05|a^m|^4 + 0.5|a^m|^2\right]$$
                                                                                              
  No requiere ninguna densidad — solo evaluar $\ell$ en los M puntos y promediar. En el       
  código, en lugar de iterar sobre las M neuronas explícitamente, se itera sobre todos los
  parámetros escalares $\theta_j$:                                                            
                                                            
  for p in self.parameters():
      pen = pen + c1 * (p ** 4).sum() + c2 * (p ** 2).sum()
      n += p.numel()                                                                          
  return pen / n
                                                                                              
  Que es exactamente $\frac{1}{N_p}\sum_j [0.05,\theta_j^4 + 0.5,\theta_j^2]$.


  -------------------------------------
  El código no calcula $|a^m|^4$ ni $|a^m|^2$ en el sentido de norma del vector $a^m$.        
                                                                                        
  En lugar de eso, eleva cada escalar individual a la 4 o al cuadrado y los suma todos:       
                                                                                              
  for p in self.parameters():                                                                 
      pen = pen + c1 * (p ** 4).sum() + c2 * (p ** 2).sum()
                                                                                              
  Es decir, para cada tensor de parámetros (W1.weight, W1.bias, W0.weight) hace:              
                                                                                              
  $$\sum_j \theta_j^4 \quad \text{y} \quad \sum_j \theta_j^2$$                                
                  
  donde $j$ recorre todos los escalares del tensor.                                           
                  
  ---                                                                                         
  Esto es distinto a lo que dice la fórmula teórica. La fórmula usa la norma euclídea de
  $a^m$:                                                                                      
                  
  $$|a^m|^2 = \sum_{k=1}^6 (a^m_k)^2 \qquad |a^m|^4 = \left(\sum_{k=1}^6 (a^m_k)^2\right)^2$$ 
                                                                                              
  Pero el código calcula:
                                                                                              
  $$\sum_{k=1}^6 (a^m_k)^2 \qquad \sum_{k=1}^6 (a^m_k)^4$$                                    
   
  La diferencia está en el cuártico: $(|a^m|^2)^2 \neq \sum_k (a^m_k)^4$ en general. El código
   usa la versión escalar por escalar, que es más fácil de implementar y tiene el mismo efecto
   regularizador — penaliza parámetros grandes — pero no es exactamente el prior teórico del  
  paper.          

-----------------------------------------------------
en Adam, que significa ese gradiente ocn respecto a J?                                      
                                                                                        
⏺ Es la derivada parcial de $J(\theta)$ respecto a cada parámetro escalar $\theta_j$:         
                                                                                              
  $$\nabla_j J = \frac{\partial J}{\partial \theta_j}$$                                       
                                                                                              
  En términos prácticos, le dice cuánto cambia la pérdida total si modificas ligeramente ese  
  parámetro concreto. Si $\partial J / \partial \theta_j > 0$, aumentar $\theta_j$ aumenta la 
  pérdida, así que hay que disminuirlo. Si es negativo, al revés.                             
                                                                                              
  ---
  Lo no trivial aquí es que $J$ depende de $\theta_j$ de forma muy indirecta:
                                                                                              
  $$\theta_j \in {W_1, W_0} \rightarrow F(x,t) \rightarrow \text{RK4 (10 pasos)} \rightarrow
  X_T \rightarrow W \cdot X_T + b \rightarrow \sigma(\cdot) \rightarrow                       
  \mathcal{L}_{\text{BCE}}$$
                                                                                              
  El gradiente tiene que propagarse hacia atrás a través de todo ese grafo — 10 pasos de RK4, 
  cada uno con 4 evaluaciones del campo $F$. PyTorch lo hace automáticamente con
  loss.backward(), que aplica la regla de la cadena a lo largo de toda esa secuencia de       
  operaciones.    

  Además, para los parámetros de MeanFieldVelocity también hay gradiente proveniente de la    
  penalización entrópica $\varepsilon \cdot \mathcal{E}(\theta)$, que se suma al de la BCE.
  Los parámetros del clasificador ($W, b$) solo reciben gradiente de la BCE.                  
                  

------------------------------------
 Backpropagation explícita a través de la ODE                                                
                                                                                              
  La regla de la cadena general                                                               
                                                                                              
  $$\frac{\partial J}{\partial \theta_j} = \frac{\partial J}{\partial \text{logit}} \cdot     
  \frac{\partial \text{logit}}{\partial X_T} \cdot \frac{\partial X_T}{\partial \theta_j}$$   
                                                                                              
  Los dos primeros factores son fáciles. El tercero — cómo $X_T$ depende de los pesos a través
   de 10 pasos de RK4 — es lo complejo.
                                                                                              
  ---                                                                                         
  Paso 1: $\partial J / \partial \text{logit}$
                                                                                              
  La derivada de BCE con sigmoide tiene una forma muy limpia:
                                                                                              
  $$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \text{logit}_i} = \sigma(\text{logit}_i)
   - y_i$$                                                                                    
                                                                                              
  Si el modelo predice bien, $\sigma(\text{logit}_i) \approx y_i$ y el gradiente es casi cero.
   Si falla, el gradiente es grande.
                                                                                              
  ---             
  Paso 2: $\partial \text{logit} / \partial X_T$
                                                
  El clasificador es $\text{logit}_i = W \cdot X_T^i + b$, con $W \in \mathbb{R}^{1 \times
  2}$. La derivada es simplemente:                                                            
   
  $$\frac{\partial \text{logit}_i}{\partial X_T^i} = W \in \mathbb{R}^{1 \times 2}$$          
                  
  Combinando con el paso 1, el gradiente que llega a $X_T$ es:                                
                  
  $$\frac{\partial J}{\partial X_T^i} = (\sigma(\text{logit}_i) - y_i) \cdot W \in            
  \mathbb{R}^{1 \times 2}$$
                                                                                              
  ---             
  Paso 3: a través de los 10 pasos de RK4
                                         
  Aquí es donde está la complejidad. Cada paso RK4 calcula:
                                                                                              
  $$x_{n+1} = x_n + \frac{dt}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$                                   
                                                                                              
  con:                                                                                        
                  
  $$k_1 = F(x_n,\ t_n)$$                                                                      
  $$k_2 = F(x_n + \tfrac{dt}{2}k_1,\ t_n + \tfrac{dt}{2})$$
  $$k_3 = F(x_n + \tfrac{dt}{2}k_2,\ t_n + \tfrac{dt}{2})$$                                   
  $$k_4 = F(x_n + dt\cdot k_3,\ t_n + dt)$$                                                   
                                                                                              
  Dado $\partial J/\partial x_{n+1}$, el gradiente hacia $x_n$ es:                            
                                                                                              
  $$\frac{\partial J}{\partial x_n} = \frac{\partial J}{\partial x_{n+1}} \cdot \frac{\partial
   x_{n+1}}{\partial x_n}$$
                                                                                              
  donde $\partial x_{n+1}/\partial x_n$ se expande por la cadena a través de $k_1, k_2, k_3,  
  k_4$. Como $k_2$ depende de $k_1$, $k_3$ depende de $k_2$, y $k_4$ depende de $k_3$, hay que
   ir hacia atrás en orden:                                                                   
                  
  $$\frac{\partial x_{n+1}}{\partial x_n} = I + \frac{dt}{6}\left(\frac{\partial k_1}{\partial
   x_n} + 2\frac{\partial k_2}{\partial x_n} + 2\frac{\partial k_3}{\partial x_n} +
  \frac{\partial k_4}{\partial x_n}\right)$$                                                  
                  
  donde por ejemplo:                                                                          
   
  $$\frac{\partial k_2}{\partial x_n} = \underbrace{\frac{\partial F}{\partial x}\bigg|{x_n + 
  \frac{dt}{2}k_1}}{\text{Jacobiano de }F} \cdot \left(I + \frac{dt}{2}\frac{\partial
  k_1}{\partial x_n}\right)$$                                                                 
                  
  Esto se repite para los 10 pasos, propagando el gradiente desde $X_T$ hasta $x_0$.          
   
  ---                                                                                         
  Paso 4: $\partial J / \partial \theta_j$ a través de $F$
                                                                                              
  En cada una de las $10 \times 4 = 40$ evaluaciones de $F$ durante el forward pass, $F$
  depende de $\theta_j$ (W1 y W0). Para cada evaluación de $F(x, t)$:                         
                  
  $$F(x,t) = W_0 \cdot \tanh(W_1 [x,t]^\top + b_1)$$                                          
                  
  $$\frac{\partial F}{\partial W_0} = h^\top, \quad h = \tanh(W_1[x,t]^\top + b_1)$$          
                  
  $$\frac{\partial F}{\partial W_1} = W_0 \cdot (1 - h^2) \cdot [x,t]^\top$$                  
                  
  El gradiente total de $J$ respecto a $W_1$ y $W_0$ es la suma de contribuciones de las 40   
  evaluaciones — cada evaluación de $F$ en cada paso de RK4 aporta su propio gradiente, y
  todos se acumulan.                                                                          
                  
  Además, para $W_1$ y $W_0$ también llega gradiente directamente desde la penalización       
  entrópica:
                                                                                              
  $$\frac{\partial \mathcal{E}}{\partial \theta_j} = \frac{1}{N_p}\left[4 \cdot               
  0.05,\theta_j^3 + 2 \cdot 0.5,\theta_j\right] = \frac{0.2,\theta_j^3 + \theta_j}{N_p}$$
                                                                                              
  ---             
  Lo que hace PyTorch
                     
  Todo esto lo calcula loss.backward() automáticamente. Durante el forward pass PyTorch
  construye un grafo computacional que registra cada operación — cada suma, multiplicación,   
  tanh — y quién depende de quién. En el backward simplemente recorre ese grafo al revés
  aplicando la regla de la cadena en cada nodo. No hay ninguna fórmula explícita programada — 
  solo el grafo y las derivadas elementales de cada operación.

 ---                                                                                                                             
  Qué hace el experimento B3                                                                                                      
                                                                                                                                  
  Objetivo: comprobar empíricamente que los parámetros aprendidos por la red tienen la forma de Gibbs que predice el paper (ec.   
  1.9).                                                                                                                           
                                                            
  La predicción teórica                                                                                                           
                                                            
  El paper demuestra que el control óptimo ν_t* bajo regularización entrópica tiene la forma:                                     
   
  $$\nu_t^*(a) \propto \exp!\Bigl(-\ell(a) - \tfrac{1}{\varepsilon}\int_A b(x,a)\cdot\nabla u_t,d\gamma_t\Bigr)$$                 
                                                            
  El prior ν^∞ es la medida de referencia que se usa en la divergencia KL, y corresponde precisamente a la parte energética:      
                                                            
  $$\nu^\infty(a) \propto e^{-\ell(a)} = \exp!\bigl(-0.05,a^4 - 0.5,a^2\bigr)$$                                                   
                                                            
  Esta es una distribución acampanada centrada en 0, con colas que caen como una gaussiana/laplaciana mezclada. El segundo término
   (el integral con ∇u_t) "deforma" esta campana según lo que la clasificación necesita.
                                                                                                                                  
  Consecuencia clave: a medida que ε → ∞, el segundo término pierde peso y ν_t* → ν^∞. Es decir, con ε muy grande los parámetros  
  deben parecerse al prior.
                                                                                                                                  
  ---                                                                                                                             
  Lo que hace el código
                                                                                                                                  
  1. Toma los parámetros entrenados de model.velocity (todos los pesos y biases de W1 y W0, aplanados en un único vector 1D de
  ~640 escalares)                                                                                                                 
  2. Dibuja el histograma empírico de esos parámetros (barras de colores = ν*)
  3. Dibuja encima la curva teórica del prior ν^∞ ∝ exp(−0.05a⁴ − 0.5a²) en blanco discontinuo                                    
  4. Añade el std de los parámetros en la esquina superior derecha                                                                
                                                                                                                                  
  ---                                                                                                                             
  Lo que muestra la figura                                  
                          
  Hay 5 paneles, uno por cada valor de ε (0, 0.001, 0.01, 0.1, 0.5):
                                                                                                                                  
  ┌────────────┬───────────────────────────────────────────────────────┬────────────┐                                             
  │     ε      │                      Histograma                       │    std     │                                             
  ├────────────┼───────────────────────────────────────────────────────┼────────────┤                                             
  │ 0          │ Distribución ancha, puede ser bimodal o irregular     │ Mayor      │
  ├────────────┼───────────────────────────────────────────────────────┼────────────┤
  │ 0.001–0.01 │ Campana más concentrada, empieza a parecerse al prior │ Intermedio │                                             
  ├────────────┼───────────────────────────────────────────────────────┼────────────┤                                             
  │ 0.1–0.5    │ Histograma muy parecido a la curva blanca             │ Menor      │                                             
  └────────────┴───────────────────────────────────────────────────────┴────────────┘                                             
                                                            
  El efecto esperado y observable es:                                                                                             
  - ε grande → std(θ) pequeño → histograma ≈ prior (curva blanca y barras se solapan bien)
  - ε = 0 → sin prior → parámetros libres, la distribución puede ser muy distinta al prior                                        
                                                                                          
  La curva blanca no cambia entre paneles (es siempre ν^∞), pero el histograma de colores sí cambia. Cuando ε es grande, los      
  parámetros "no se alejan mucho de 0" porque la regularización los empuja hacia ν^∞, que tiene su máximo exactamente en a=0.                                                                                                                  
