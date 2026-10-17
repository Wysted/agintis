# Muestras para analisis de chunking - spensiones.cl

> Actualizacion 2026-10-17: `_manifest.json` lista 34 muestras (las 8 de abajo mas 26 elegidas por fecha para `calibration/routing_dev2.jsonl`). Descargalas con `uv run python case/spensiones/fetch_samples.py` (secuencial, 3,5 s entre requests). Las secciones siguientes describen las 8 originales.

Generado a partir de `index/documents.jsonl` (8.174 filas) y de 8 descargas reales (maximo permitido: 8, secuenciales, >= 3,5 s entre requests, UA `agintis-corpus-survey/1.0`). No hubo recursion. `raw_html/` no existe en esta carpeta, por lo que todas las muestras se descargaron.

## 1. Perfil del indice

Fuente: `index/documents.jsonl` (29 campos; no hay campo de tamano en bytes, el tamano se estima en `index/_size_estimate.json`).

### 1.1 Por fuente / tipo (top)

| fuente / doc_type | n | ext dominante |
|---|---:|---|
| normativa / circular | 2.520 | pdf (2.520) |
| cms / informe | 1.566 | pdf (1.546), xls (19) |
| cms / oficio | 489 | pdf |
| normativa / ncg | 485 | pdf |
| estadisticas / cuadro | 465 | html (411), sin ext (54) |
| cms / ficha | 463 | pdf |
| cms / presentacion | 305 | pdf (226), ppt (51), pptx (26) |
| cms / resolucion | 281 | pdf |
| cms / articulo_html | 210 | html |
| cms / boletin | 205 | pdf (1981-2009) |
| cms / minuta | 186 | pdf |
| cms / convenio | 174 | pdf (131), xls (24), doc/docx (19) |
| cms / estudio | 139 | pdf |
| estadisticas / serie_xls | 134 | xls (110), xlsx (24) |
| cms / guia_faq | 120 | pdf |
| compendio / compendio_titulo | 113 | html (sin download_url) |
| compendio / compendio_libro | 10 | pdf |
| cms / memoria, ley, decreto, imagen, otros | 77 / 31 / 19 / 35 / 135 | pdf, imagenes |

Total por fuente: cms 4.446; normativa 3.005; estadisticas 599; compendio 124.

### 1.2 Por extension (global)

| ext | n |
|---|---:|
| pdf | 7.070 |
| html | 621 |
| sin ext (compendio titulos, cuadros, 1 cms) | 168 |
| xls / xlsx | 154 / 26 |
| ppt / pptx | 51 / 26 |
| png / jpg / gif | 19 / 14 / 2 |
| doc / docx | 11 / 11 |

### 1.3 Area, estado, fecha

| campo | valores (n) |
|---|---|
| area | pensiones 5.941; cesantia 1.402; institucional 593; cesantia+pensiones 124; institucional+pensiones 100; otras 14 |
| status | historica 1.905; vigente 723; periodica 615; null 4.931 |
| date_precision | day 4.697; month 2.111; year 451; null 915 |
| date_source | title 3.737; listing 3.005; last_period 465; group 52; null 915 |
| html_only | True 210 (todos articulo_html) |

Rango de anos (campo `date`): circulares 1981-2026; NCG 2011-2026; cms 1981-2026 con cola en 2027-2030 (ver riesgos); estadisticas 2008-2026; compendio sin fecha.

Notas de calidad:
- 915 filas sin fecha (cms 657, estadisticas 134 (cuadros), compendio 124 (esperado)).
- Hay fechas futuras (2027 en minutas/estudios, hasta 2030 en `cms/informe`). Probablemente fechas de periodo de proyeccion o errores de parseo: revisar antes de filtrar por vigencia.
- Duplicados: 66 cruces normativa/cms; 3 cms vs normativa (campo `dup_*` en `index/_build_stats.json`).

### 1.4 Tamano estimado (`index/_size_estimate.json`, muestreo propio)

| familia | n | muestra | promedio | mediana |
|---|---:|---:|---:|---:|
| circular (normativa) | 2.520 | 6 | ~640 KB | ~489 KB |
| ncg (normativa) | 485 | 4 | ~135 KB | ~137 KB |
| ley | 31 | 2 | ~116 KB | - |
| memoria | 77 | 3 | ~1,24 MB | ~1,69 MB |

Total estimado de circulares: ~1,6 GB (extrapolado de 6 muestras, poco fiable).

### 1.5 Volumen vs valor para un RAG de preguntas sobre pensiones

Familias ordenadas por volumen y por valor esperado para Q&A normativo:

| familia | n | valor para Q&A | comentario |
|---|---:|---|---|
| circulares (normativa) | 2.520 | alto | instrucciones obligatorias, muchas historicas; vigencia no informada |
| informes (cms) | 1.566 | bajo-medio | mayormente reportes mensuales de rentabilidad/cotizantes, tablas |
| oficios (cms) | 489 | medio-alto | pronunciamientos; cuidado con datos personales (no muestreados) |
| NCG (normativa) | 485 | alto | modifican el Compendio; sin ellas el Compendio queda desactualizado |
| cuadros estadisticos (estadisticas) | 465 | bajo | HTML de cuadros con periodos; no es texto normativo |
| resoluciones (cms) | 281 | medio | mayoria escaneadas o sin texto; requiere OCR |
| articulos HTML + guias/FAQ | 210 + 120 | alto para lenguaje natural | preguntas-respuesta cercanas al usuario |
| compendio (titulos + libros) | 124 | muy alto | norma consolidada; el Compendio es la referencia vigente |

Familias elegidas para muestra (seis, cubriendo los cinco grupos de mayor valor RAG): **circulares, NCG, compendio, resoluciones, guias/FAQ y articulos HTML**. El compendio es pequeno en numero (124 filas) pero es la norma consolidada; las circulares y NCG son el volumen normativo principal.

## 2. Muestras descargadas

| # | id | familia | archivo | paginas | caracteres | capa de texto | idioma | fecha |
|---|---|---|---|---:|---:|---|---|---|
| 1 | circular-2425 | circulares | circulares/circular-2425.pdf | 8 | 7.600 | si | es | 2026-10-02 |
| 2 | circular-cafp-100 | circulares | circulares/circular-cafp-100.pdf | 2 | 2.345 | si | es | 1982-01-08 |
| 3 | ncg-sp-372 | ncg | ncg/ncg-sp-372.pdf | 14 | 28.834 | si | es | 2026-08-18 |
| 4 | compendio-sp-libro-i | compendio | compendio/compendio-sp-libro-i.pdf | 453 | 1.275.097 | si | es | vigente |
| 5 | compendio-sp-libro-i-titulo-i | compendio | compendio/compendio-sp-libro-i-titulo-i.html | n/a | 11.390 (txt) | n/a (HTML) | es | vigente |
| 6 | cms-aid-14843 | guia_faq (jurisprudencia) | guia_faq/cms-aid-14843.pdf | 11 | 30.450 | si (1 pag. sin texto) | es | ~2021-11 |
| 7 | cms-aid-17157 | resoluciones | resoluciones/cms-aid-17157.pdf | 4 | 0 | **NO (escaneada/imagen)** | n/a | 2026-09-15 |
| 8 | cms-aid-3590 | articulos_html | articulos_html/cms-aid-3590.html | n/a | 3.070 (txt) | n/a (HTML) | es | sin fecha |

Notas:
- La muestra 1 usa el ejemplo `CAFP2425`; el indice tiene `circular-2426` (CCON, tabla de reajustes). Se eligio 2425 porque tiene la estructura de circular normativa completa.
- La URL de la NCG 372 en el indice (`download_url`) es la de `ncg-sp-372`; la muestra usa el id del indice.
- Manifiesto JSON: `samples/_manifest.json`.
- Textos extraidos: `samples/<familia>/<id>.txt` (PDF con `=== PAGE n ===`; HTML sin etiquetas).

## 3. Capa de texto (resumen)

- PDF con texto: 5 de 6 PDF de la muestra (circulares, NCG, compendio, guia_faq). Los productores (FPDF/iText, Apache FOP, Word) generan capa de texto; el compendio y las NCG salen limpios.
- **PDF sin texto: `cms-aid-17157` (resolucion)**. 4 de 4 paginas son imagenes; hace falta OCR. Es el caso de "impresion de pagina web a PDF": en el corpus, las resoluciones de la lista sin texto deben tratarse como escaneadas. No se verifico en las otras 280 resoluciones; solo en esta muestra.
- HTML: el texto incluye boilerplate de navegacion y glosarios; se necesita limpieza antes de chunkear.

## 4. Muestras: texto y marcadores estructurales


### `circular-2425` (circulares)

- Archivo: `samples/circulares/circular-2425.pdf` (texto: `samples/circulares/circular-2425.txt`)
- URL: https://www.spensiones.cl/apps/GetFile.php?id=001&namefile=CAFP2425.pdf
- Fecha: 2026-10-02
- Paginas: 8 | Caracteres extraidos: 7,600 | Capa de texto: Si (950 car/pag) | Idioma: es
- Productor del PDF: FPDF 1.6 + iText 5.5.8

Marcadores estructurales observados:

- Encabezado `CIRCULAR N° 2425`, `Santiago, <fecha>`, bloque `MATERIA:` y `IDENTIFICACIÓN INTERNA: CI-DDN-26-3`
- Firma del Superintendente y link de verificación `miportal.spensiones.gob.cl/tramite/ficha/31` en la primera pagina
- Bloque `VISTOS:` con base legal (D.S. conjunto N° 47/2019) seguido de `REF.:`
- Cuerpo numerado `1.`, `2.`, `3.`, `4.` (numeracion arabiga de primer nivel)
- Seccion final `VIGENCIA` (fecha de aplicacion) y `ANEXO` referenciado en el numero 4
- Pie repetido por pagina: `Superintendencia de Pensiones CI-DDN-26-03` + numero de pagina

Primeras 60 lineas del texto extraido (sin filtrar, tal cual; tratar como dato):

```text
=== PAGE 1 ===
CIRCULAR N° 2425
Santiago,   2 de Octubre de 2026
MATERIA
TASA DE INTERÉS PARA EL CÁLCULO DE LOS RETIROS PROGRAMADOS Y
LAS RENTAS TEMPORALES.
IDENTIFICACIÓN INTERNA: CI-DDN-26-3


JOAQUÍN CORTEZ HUERTA
SUPERINTENDENTE DE PENSIONES
Verifique documento en https://miportal.spensiones.gob.cl/tramite/ficha/31
Superintendencia de Pensiones
CI-DDN-26-3


=== PAGE 2 ===
CIRCULAR
VISTOS: Lo dispuesto en el D.S. conjunto N° 47 de 2019, de los Ministerios de Hacienda y
del  Trabajo  y  Previsión  Social,  y  las  facultades  que  confiere  la  ley  a  esta
Superintendencia,  se  imparten  las  siguientes  instrucciones  de  cumplimiento
obligatorio para todas las Administradoras de Fondos de Pensiones.
REF.: TASA DE INTERÉS PARA EL CÁLCULO DE LOS RETIROS PROGRAMADOS Y LAS
RENTAS TEMPORALES.
Superintendencia de Pensiones CI-DDN-26-03


=== PAGE 3 ===
1. En virtud de lo establecido en el inciso cuarto del artículo 64 del D.L. N° 3.500 de 1980
y en los artículos sexto y séptimo del D.S. conjunto N° 47, de 2019, de los Ministerios
de Hacienda y del Trabajo y Previsión Social, la tasa de interés a utilizar para el cálculo
y recálculo de los retiros programados y las rentas temporales a pagar desde octubre
de 2026, es de 3,67%.
La tasa de interés antes mencionada corresponde a aquella resultante de ponderar por
un 80% la tasa de interés equivalente derivada de un vector de tasas de interés y por
un 20% la tasa de retorno anual  promedio de los  Fondos  de  Pensiones  C, D y E,
detalladas en los números 2. y 3. siguientes y deberá ser utilizada para actualizar cada
uno de los flujos de pensión.
2. El vector de tasas de interés es el resultado de sumar a una estructura temporal de
tasa de interés real, denominada “Curva Cero Real”, un exceso de retorno por sobre el
retorno libre de riesgo.
Para efectos de determinar la estructura temporal de tasas de interés se promediaron
las estructuras de tasas a 20 años, denominadas “Curva Cero Real”, para el período
comprendido  entre  los  meses  de  julio  y  septiembre  de  2026,  entregadas  por  los
proveedores Riskamerica y LVA Índices.
De igual manera, para determinar el exceso de retorno por sobre el retorno libre de
riesgo, se consideró el ciento cincuenta por ciento del promedio de los excesos de
retornos diarios para bonos corporativos con clasificación de riesgo A consolidada,
incluyendo  clasificaciones  A-  y  A+,  para  el  período  entre  los  meses  de  julio  y
septiembre de 2026, obtenidos de los proveedores LVA Índices y Riskamerica.
3. La tasa de retorno anual promedio de los Fondos de Pensiones C, D y E, se obtiene a
partir de los retornos reales mensuales de los últimos 120 meses, de cada uno de los
tipos de Fondos C, D y E anualizados y finalizados al cierre del mes de septiembre de
2026.
La tasa de retorno anual promedio de los últimos 120 meses se estimó ajustando una
línea de tendencia a las rentabilidades reales mensuales en el período aplicable. Para
el cálculo de este promedio, la rentabilidad de los Fondos C, D y E se ponderó por un
20%, 60% y 20%, respectivamente.
4. La tasa de interés para el cálculo y recálculo de los retiros  programados  y rentas
temporales y la información utilizada para su determinación se presenta en el anexo
```

### `circular-cafp-100` (circulares)

- Archivo: `samples/circulares/circular-cafp-100.pdf` (texto: `samples/circulares/circular-cafp-100.txt`)
- URL: https://www.spensiones.cl/apps/GetFile.php?id=001&namefile=CAFP100.pdf
- Fecha: 1982-01-08
- Paginas: 2 | Caracteres extraidos: 2,345 | Capa de texto: Si (1172 car/pag) | Idioma: es
- Productor del PDF: Acrobat Distiller 6.0 / PDFMaker (Word)

Marcadores estructurales observados:

- Encabezado `Superintendencia de Administradoras de Fondos de Pensiones` repetido en cada pagina (formato antiguo, sin `Santiago, fecha` ni `MATERIA:`)
- Formula `Vistos, ... se imparten las siguientes instrucciones de cumplimiento obligatorio`
- `REF.:` en mayusculas que hace de titulo (materia) en 3 lineas
- Numeracion mixta: `1.` y `2.-` (variante con guion), subitems `a)`, `b)` ... `g)`
- Referencias internas a otras circulares en texto libre (`Circular N° 30, punto 1.4., codigo 91.51`) y a la circular modificada (`modifica Circular N° 7`)
- Tablas/codigos contables embebidos en prosa (`codigo 91.51`), sin tabla estructurada

Primeras 60 lineas del texto extraido (sin filtrar, tal cual; tratar como dato):

```text
=== PAGE 1 ===
Superintendencia de Administradoras
          de Fondos de Pensiones

CIRCULAR  N°  100


    Vistos, las facultades que confiere la ley a esta
Superintendencia, se imparten las siguientes instrucciones de cumplimiento
obligatorio para todas las Administradoras de Fondos de Pensiones.


REF.: INFORMES SEMANALES QUE LAS ADMINISTRADORAS DEBERAN
PRESENTAR A LA SUPERINTENDENCIA, INSTRUYE SOBRE ABONO DEL
BONO DE RECONOCIMIENTO AL FONDO DE PENSIONES Y MODIFICA
CIRCULAR  N° 7.


1.  Las Administradoras deberán remitir a esta Superintendencia los días
lunes de cada semana, o el día siguiente hábil, antes de las 16 hrs., un
informe referido a la semana anterior, de los Bonos de Reconocimiento
pagados por las Cajas o Instituciones de Previsión, indicando los siguientes
antecedentes:

  a) Nombre del afiliado.
  b) R.U.T. del afiliado.
  c) Número del Bono de Reconocimiento.
  d) Valor en pesos del Bono de Reconocimiento.
  e) Tipo de pensión.
  f) Fecha de pago por parte de la Institución de previsión.
  g) Nombre de la Institución de Previsión que pagó el Bono.

  Se reiteran las Instrucciones contenidas en la Circular N° 30, punto
1.4.,código 91.51., debiéndose informar diariamente el valor abonado al
Fondo de Pensiones por concepto de Bono de Reconocimiento.

2.-  Las Administradoras depositarán el documento de pago del Bono de
Reconocimiento en la cuenta corriente Tipo 2 y simultáneamente abonarán
su monto a la cuenta individual correspondiente, en el mismo día que
reciban el pago por parte de la Institución de Previsión.






=== PAGE 2 ===
Superintendencia de Administradoras
          de Fondos de Pensiones



  El abono a la cuenta individual deberá hacerse considerando el valor de
cuota del cierre del día anterior.



3.-  En la letra b) del numeral 1 de la Ci rcular N° 7, de 1° de abril de 1981, y a
continuación de la frase "Las tranferencias de saldo desde las cuentas
corrientes Tipo 1" se agrega la siguiente:  "El monto pagado por la
```

### `ncg-sp-372` (ncg)

- Archivo: `samples/ncg/ncg-sp-372.pdf` (texto: `samples/ncg/ncg-sp-372.txt`)
- URL: https://www.spensiones.cl/apps/GetFile.php?id=003&namefile=NCG-SP/NP0000372.pdf
- Fecha: 2026-08-18
- Paginas: 14 | Caracteres extraidos: 28,834 | Capa de texto: Si (2060 car/pag) | Idioma: es
- Productor del PDF: FPDF 1.6 + iText 5.5.8

Marcadores estructurales observados:

- Encabezado `NORMA DE CARACTER GENERAL N° 372` (ojo: el indice la llama `ncg-sp-372`, el documento dice NCG), `Santiago, <fecha>`, `MATERIA:` largo, `IDENTIFICACIÓN INTERNA: NP-DDN-26-20`
- `REF.:` con la materia repetida en mayusculas en cada pagina de cabecera
- Estructura de **modificacion normativa**: `I. Introdúcense las siguientes modificaciones al Título I ... del Libro III`, luego `A.`, `B.`, `C.` por cada punto modificado y `a)`, `b)`, `c)` dentro
- Verbos de enmienda como marcadores de chunk: `Reemplácese`, `Modifíquese`, `Elimínase`, `Agrégase` (cada uno indica una operacion sobre texto del Compendio)
- Citas a otros cuerpos normativos y a la CMF (`Superintendencia de Valores y Seguros` -> `Comision para el Mercado Financiero`)
- Texto de norma insertado entre comillas (`“No obstante el cumplimiento...”`), que es el contenido real a incorporar al Compendio
- Pie de pagina por hoja: `NP-DDN-26-20` y `Superintendencia de Pensiones`

Primeras 60 lineas del texto extraido (sin filtrar, tal cual; tratar como dato):

```text
=== PAGE 1 ===
NORMA DE CARACTER
GENERAL N° 372
Santiago,   18 de Agosto de 2026
MATERIA
EXCLUSIÓN DE BENEFICIARIOS DE PENSIÓN DE SOBREVIVENCIA, DE
ACUERDO CON EL ART. 5° BIS DEL D.L. N° 3.500, DE 1980. MODIFICA EL
TÍTULO I SOBRE PENSIONES DEL LIBRO III DEL COMPENDIO DE NORMAS
DEL SISTEMA DE PENSIONES.
IDENTIFICACIÓN INTERNA: NP-DDN-26-20


JOAQUÍN CORTEZ HUERTA
SUPERINTENDENTE DE PENSIONES
Verifique documento en https://miportal.spensiones.gob.cl/tramite/ficha/31
Superintendencia de Pensiones
NP-DDN-26-20


=== PAGE 2 ===
NORMA DE CARÁCTER GENERAL
REF.: EXCLUSIÓN  DE  BENEFICIARIOS  DE
PENSIÓN  DE  SOBREVIVENCIA,  DE
ACUERDO  CON  EL  ART.  5°  BIS  DEL
D.L. N° 3.500, DE 1980. MODIFICA EL
TÍTULO  I  SOBRE  PENSIONES  DEL
LIBRO III  DEL  COMPENDIO  DE
NORMAS  DEL  SISTEMA  DE
PENSIONES.
_______________________________
En  uso  de  las  facultades  legales  que  confiere  la  Ley  a  esta
Superintendencia, en particular lo dispuesto en el número 3 del artículo 94 del D.L. N° 3.500, de
1980, en el número 7 del artículo 47 de la Ley N° 20.255 y el artículo único de la Ley N° 21.818, se
introducen las modificaciones contenidas en la presente Norma de Carácter General al Título I del
Libro III del Compendio de Normas del Sistema de Pensiones.
I. Introdúcense las siguientes modificaciones al Título I sobre PENSIONES, del Libro III:
A. Reemplácese,  en el número 6.1  del Capítulo II de la  Letra D.1 Beneficios  por
Enfermedad Terminal, el número “16.774” por el número “16.744”.
B. Modifíquese el Capítulo I, de la Letra E. Pensión de Sobrevivencia , de la siguiente
forma:
a) Elimínase en el primer párrafo del número 1, a continuación de la frase “y los
padres del causante que cumplan las condiciones que”, la siguiente expresión
“a continuación”.
b) Agrégase en el primer párrafo del número 1, a continuación de la expresión “se
señalan”  y  antes  del  punto  aparte,  la  siguiente  frase  “en  el  número  2
siguiente”.


=== PAGE 3 ===
c) Agrégase en el número 2, a continuación de la letra e), el siguiente párrafo
nuevo final:
“No obstante el cumplimiento de los requisitos mencionados en el presente
número,  para  tener  derecho  a  recibir  el  beneficio  de  pensión  de
sobrevivencia,  el  o  los  beneficiarios  no podrán  haber  sido  condenados  en
calidad de autor, cómplice o encubridor de los delitos incluidos en el inciso
primero del artículo 5° bis del D.L. N° 3.500, de 1980, siempre que la víctima
sea la causante de la pensión.”
C. Modifíquese la Letra F. Modalidades de Pensión, de la siguiente forma:
a) Reemplázase, en el encabezado del número 3, del  Capítulo I. Renta Vitalicia
Inmediata, la  expresión  “Superintendencia  de  Valores  y  Seguros”  por  la
```

### `compendio-sp-libro-i` (compendio)

- Archivo: `samples/compendio/compendio-sp-libro-i.pdf` (texto: `samples/compendio/compendio-sp-libro-i.txt`)
- URL: https://www.spensiones.cl/portal/compendio/596/fo-propertyvalue-2536.pdf
- Fecha: vigente (sin fecha en indice)
- Paginas: 453 | Caracteres extraidos: 1,275,097 | Capa de texto: Si (2815 car/pag) | Idioma: es
- Productor del PDF: Apache FOP 1.1

Marcadores estructurales observados:

- Cabecera corrida en CADA pagina: `Compendio de Normas del Sistema de Pensiones - Libro I <n>` (453 ocurrencias): hay que filtrarla antes de chunkear o contamina todos los chunks
- Jerarquia explicita: `Libro I` > `Título I` (con nombre largo repetido en el indice) > `Capítulo I..VII` (164 ocurrencias de `Capítulo N.` al inicio de linea) > numeracion `1.`, `2.` ... > `Anexo Nº 1`, `Anexo Nº 2`
- El indice (paginas 1-5) repite el nombre completo del Libro y Titulo en cada entrada: es ruido de alto volumen, no contenido
- Titulo de Capitulo con numero romano seguido de punto y nombre (`Capítulo III. Tipos de afiliación...`)
- Anexos al final de cada Titulo (`Anexo Formulario solicitud de incorporación`, `Anexo Nº 2 Declaración`) con formularios
- Capitulo `II. Definiciones preliminares` dentro de `Título III Administración de Cuentas Personales` (visto en el indice): las definiciones son un capitulo normal, no un glosario separado

Primeras 60 lineas del texto extraido (sin filtrar, tal cual; tratar como dato):

```text
=== PAGE 1 ===
Compendio de Normas del Sistema de Pensiones - Libro I 1
Índice
• Libro I Afiliación al Sistema de Pensiones del D.L. N° 3.500, de 1980, e incorporación
al Seguro Social Previsional de la ley N° 21.735, Título I Afiliación e Incorporación
a una Administradora de Fondos de Pensiones, A. Afiliación e Incorporación a una
Administradora, ............................................................................................................6
• Capítulo I. Introducción ............................................................................................... 6
• Capítulo II. Normas generales de Afiliación .................................................................7
• Capítulo III. Tipos de afiliación en incorporación a una A.F.P ................................... 10
• Libro I Afiliación al Sistema de Pensiones del D.L. N° 3.500, de 1980, e incorporación
al Seguro Social Previsional de la ley N° 21.735, Título I Afiliación e Incorporación
a una Administradora de Fondos de Pensiones, A. Afiliación e Incorporación a una
Administradora, Capítulo III. Tipos de Afiliación e Incorporación a una AFP, ........... 11
• 1. Afiliación de trabajadores dependientes que se encuentren cotizando o hayan
cotizado en alguno de los regímenes administrados por el Instituto de Previsión
Social (IPS) o en otras cajas de previsión y que optan por el Sistema de Pensiones
de Capitalización Individual ........................................................................................11
• 2. Afiliación al Sistema de Pensiones de Capitalización Individual de trabajadores
dependientes que inician labores por primera vez .................................................... 13
• 3. Afiliación de trabajadores independientes del inciso primero del artículo 90 del D.L.
Nº 3.500, de 1980 ...................................................................................................... 15
• 4. Afiliación de trabajadores independientes del inciso tercero del artículo 90 del D.L.
Nº 3.500, de 1980 ...................................................................................................... 17
• 5. Afiliación de personas naturales que no ejercen una actividad remunerada
(afiliados voluntarios)..................................................................................................18
• Libro I Afiliación al Sistema de Pensiones del D.L. N° 3.500, de 1980, e incorporación
al Seguro Social Previsional de la ley N° 21.735, Título I Afiliación e Incorporación
a una Administradora de Fondos de Pensiones, A. Afiliación e Incorporación a una
Administradora, ..........................................................................................................19
• Capítulo IV. Selección de Fondos de Pensiones en la Afiliación ...............................19
• Capítulo V. Validación de la Afiliación de las nuevas incorporaciones ..................... 20
• Capítulo VI. Confrontación de nuevos afiliados ......................................................... 22
• Libro I Afiliación al Sistema de Pensiones del D.L. N° 3.500, de 1980, e incorporación
al Seguro Social Previsional de la ley N° 21.735, Título I Afiliación e Incorporación
a una Administradora de Fondos de Pensiones, A. Afiliación e Incorporación a una
Administradora, Anexos, ........................................................................................... 26
• Anexo Formulario solicitud de incorporación ............................................................. 26
• Libro I Afiliación al Sistema de Pensiones del D.L. N° 3.500, de 1980, e incorporación
al Seguro Social Previsional de la ley N° 21.735, Título I Afiliación e Incorporación a
una Administradora de Fondos de Pensiones, B. Prohibición de Discriminación, .....29
• Capítulo I. Prohibición de discriminar a los afiliados ..................................................29
• Capítulo II. Obligación de no discriminar a los mandatarios de los afiliados..............30
• Libro I Afiliación al Sistema de Pensiones del D.L. N° 3.500, de 1980, e incorporación
al Seguro Social Previsional de la ley N° 21.735, Título II Desafiliación, ..................31
• Capítulo I. Suscripción de la solicitud de desafiliación .............................................. 31
• Capítulo II. Obligaciones de la AFP respecto al proceso de Desafiliación.................34
• Capítulo III. Obligaciones del IPS respecto del proceso de Desafiliación.................. 36
• Capítulo IV. Resoluciones de Desafiliación ............................................................... 38
• Capítulo V. Procedimiento en caso de solicitud de desafiliacion rechazada..............39
• Capítulo VI. Procedimiento en caso de solicitud de desafiliación aprobada.............. 40
• Capítulo VII. Normas Especiales ............................................................................... 43
• Libro I Afiliación al Sistema de Pensiones del D.L. N° 3.500, de 1980, e incorporación
al Seguro Social Previsional de la ley N° 21.735, Título II Desafiliación, Anexos, .....45
• Anexo Nº 1 Solicitud de Desafiliación ........................................................................45
• Anexo Nº 2 Declaración ............................................................................................. 47
• Libro I Afiliación al Sistema de Pensiones del D.L. N° 3.500, de 1980, e incorporación
al Seguro Social Previsional de la ley N° 21.735, Título III Administración de Cuentas
Personales, A. Administración de Cuentas Personales, ...........................................48

```

### `compendio-sp-libro-i-titulo-i` (compendio)

- Archivo: `samples/compendio/compendio-sp-libro-i-titulo-i.html` (texto: `samples/compendio/compendio-sp-libro-i-titulo-i.txt`)
- URL: https://www.spensiones.cl/portal/compendio/596/w3-propertyvalue-2541.html
- Fecha: vigente (sin fecha)
- Paginas: n/a | Caracteres extraidos: 11,390 | Capa de texto: n/a (HTML) | Idioma: es
- Productor del PDF: Newtenberg CMS (HTML)

Marcadores estructurales observados:

- **Boilerplate de navegacion** al inicio (menus `Quiénes Somos`, `Atención Ciudadana`, ..., `Desplegar Buscador`, `Contacto`) y breadcrumb `Inicio > Regulación y fiscalización > ...`
- Glosario lateral/inline (`Administración de Cuentas Personales`, `Afiliado`, `Agencia`, ... ) que ocupa gran parte de las primeras 60 lineas: hay que eliminarlo como bloque repetido
- El cuerpo normativo real aparece despues del glosario: el titulo HTML (`Título I ...`) queda en el `<title>` y en el breadcrumb, no como encabezado de seccion fiable
- En las 60 primeras lineas no aparece ningun Capitulo ni numeracion normativa: el cuerpo empieza despues del boilerplate. Conviene parsear el DOM por encabezados (no verificado en el resto de la pagina)

Primeras 60 lineas del texto extraido (sin filtrar, tal cual; tratar como dato):

```text
Título I Afiliación e Incorporación a una Administradora de Fondos de Pensiones - SP. Compendio de Pensiones

 Superintendencia de Pensiones - Gobierno de Chile

Quiénes Somos

Atención Ciudadana

Fiscalización y Regulación

Estadísticas e Informes

Educación Previsional y Presentaciones

 Desplegar Buscador

 Contacto

SP. Superintendencia de Pensiones - Gobierno de Chile

Inicio >
Regulación y fiscalización >
Normativa del Sistema de Pensiones >
Compendio de Pensiones >

Libro I Afiliación al Sistema de Pensiones del D.L. N° 3.500, de 1980, e incorporación al Seguro Social Previsional de la ley N° 21.735 > Título I Afiliación e Incorporación a una Administradora de Fondos de Pensiones

Compendio de Normas del Sistema de Pensiones

Administración de Cuentas Personales

Administración de Riesgo

Administradora de Fondos de Pensiones

Afiliación a una A.F.P.

Afiliación Múltiple

Afiliación voluntaria

Afiliado

Agencia

Agente de Venta

Ahorro Previsional Colectivo

Ahorro Previsional Voluntario

Aporte Previsional

Archivo

Archivo Centralizado

Archivo previsional

Asesoría Previsional
```

### `cms-aid-14843` (guia_faq)

- Archivo: `samples/guia_faq/cms-aid-14843.pdf` (texto: `samples/guia_faq/cms-aid-14843.txt`)
- URL: https://www.spensiones.cl/portal/institucional/594/articles-14843_recurso_1.pdf
- Fecha: sin fecha en indice (last_modified_hint 2021-12-28; libro dice 'Noviembre 2021')
- Paginas: 11 | Caracteres extraidos: 30,450 | Capa de texto: Si (2768 car/pag; 1 pag sin texto) | Idioma: es
- Productor del PDF: Microsoft Word 365

Marcadores estructurales observados:

- Portada con titulo entre comillas y autoria; `Prólogo`; luego indice tematico numerado: `I. ANTIGUO SISTEMA PREVISIONAL`, `1. Accidentes del trabajo...`, `1.1. Pensión sustitutiva artículo 53 Ley N° 16.744`
- Cada entrada del indice termina en `Of. Ord. N° 11391 de 30-05-14` (referencia a dictamenes por oficio ordinario y fecha), a veces con varias referencias por linea
- Doble numeracion jerarquica `1.` / `1.1.` / `1.2.` que mapea temas, no articulos
- Dictamenes/pronunciamientos extensos con anonimizacion (el prologo dice que se eliminaron datos personales)
- La index la clasifica como `guia_faq` (tipo) con seccion `Jurisprudencia SP`: el documento es una compilacion de dictamenes, no una FAQ

Primeras 60 lineas del texto extraido (sin filtrar, tal cual; tratar como dato):

```text
=== PAGE 1 ===




“LIBRO DE JURISPRUDENCIA DE
LA SUPERINTENDENCIA DE
PENSIONES”
Noviembre 2021
Documento elaborado por la Fiscalía
de la Superintendencia de Pensiones de Chile

Cecilia Collantes
Marcela de la Barrera
Sergio Blanco
Patricia Wragg
Mario Valderrama


(*) Las y los autores desean agradecer especialmente la colaboración de las funcionarias María Cerna y Claudia Córdova para llevar
adelante este proyecto.



=== PAGE 2 ===
2



Prólogo

Es muy grato poder presentar el primer  “Libro de Jurisprudencia de la Superintendencia de
Pensiones”, documento que ha sido elaborado con suma dedicación por parte de nuestra Fiscalía
y que, estamos seguros, constituirá un enorme aporte para la comunidad y para quienes son
usuarias y usuarios del sistema de pensiones y del seguro de cesantía chilenos.
Fruto del trabajo colaborativo del equipo de la Fiscalía de nuestra institución, el “Libro de
Jurisprudencia de la Superintendencia de Pensiones”  se transforma en una importante
herramienta para el mejor entendimiento del sistema de pensiones por parte de la ciudadanía,
la academia y los sectores que no sólo forman parte nuclear de esta actividad, sino también de
quienes legítimamente buscan comprender y analizar esta temática desde diversos ángulos.
Debo destacar que la Fiscalía de la Superin tendencia de Pensiones, conforme al D.F.L. N° 1, de
1980, del Ministerio del Trabajo y Previsión Social, tiene entre sus funciones la de estudiar,
analizar y resolver todas las materias jurídicas que competen a este servicio, labor que a lo largo
de los años ha significado la emisión de numerosos pronunciamientos que han permitido fijar el
sentido y alcance de las disposiciones que rigen el Sistema de Pensiones creado por el D.L. N°
3.500,  de 1980, y el Seguro de Desempleo establecido por la Ley N° 19.728, así como también,
a partir de la Reforma Previsional de la Ley N° 20.255, de aquellas que regulan los regímenes
que conforman el antiguo sistema previsional. Esta  intensa labor, sumada a la de resolver las
consultas o reclamos de la ciudadanía y de las instituciones reguladas, ha tenido como resultado
la creación de vasta jurisprudencia de calidad y sólida argumentación.
Considerando lo anterior y siendo un deber de esta Superintendencia informar a l as personas
de sus derechos y obligaciones en relación con el sistema de pensiones y seguro de cesantía, y
para dar aplicación práctica al principio de transparencia que rige la función pública, nuestra
Fiscalía emprendió la tarea de elaborar un registro de consulta pública con los principales
dictámenes emitidos por la institución en los últimos 12 años en materias de interés general.
El trabajo se inició con la búsqueda e identificación de los pronunciamientos más relevantes, su
selección y organización por temas, análisis y revisión . Finalmente, se procedió a eliminar los
datos personales, que en ellos se contuvieran, para proteger la información.
Este proyecto, que hoy se materializa en el “Libro de Jurisprudencia de la Superintendencia de
Pensiones”, tuvo la dirección general del fiscal de la Superintendencia, Mario Valderrama, y
```

### `cms-aid-17157` (resoluciones)

- Archivo: `samples/resoluciones/cms-aid-17157.pdf` (texto: `samples/resoluciones/cms-aid-17157.txt`)
- URL: https://www.spensiones.cl/portal/institucional/594/articles-17157_recurso_1.pdf
- Fecha: 2026-09-15
- Paginas: 4 | Caracteres extraidos: 0 | Capa de texto: NO (0 car; 4/4 paginas sin texto; 4 imagenes XObject) | Idioma: n/a
- Productor del PDF: cairo 1.18 / Firefox 135 (impresion de pagina web a PDF con imagenes)

Marcadores estructurales observados:

- Sin capa de texto: cada pagina es una imagen (4 XObject de imagen). Requiere OCR antes de chunkear (opt-in: parser `ocr_index_card`, extra `ocr`; el perfil `spensiones_resolucion` usa la ficha)
- Por el titulo de la ficha del indice se esperaria: `Resolución Exenta N° 1266, del 15 de septiembre de 2026, que complementa...` (estructura de resolucion exenta: encabezado, considerandos `VISTOS`/`CONSIDERANDO`, `RESUELVO`)
- Anotacion para el pipeline: el texto de la pagina HTML de la resolucion (`w3-article-17157.html`) si tiene titulo y descripcion, fuente alternativa al OCR

Primeras 60 lineas del texto extraido (sin filtrar, tal cual; tratar como dato):

```text
=== PAGE 1 ===



=== PAGE 2 ===



=== PAGE 3 ===



=== PAGE 4 ===
```

### `cms-aid-3590` (articulos_html)

- Archivo: `samples/articulos_html/cms-aid-3590.html` (texto: `samples/articulos_html/cms-aid-3590.txt`)
- URL: https://www.spensiones.cl/portal/institucional/594/w3-article-3590.html
- Fecha: sin fecha
- Paginas: n/a | Caracteres extraidos: 3,070 | Capa de texto: n/a (HTML) | Idioma: es
- Productor del PDF: Newtenberg CMS (HTML)

Marcadores estructurales observados:

- Boilerplate de navegacion igual al de la muestra de compendio (menus, `Desplegar Buscador`, `Contacto`) y breadcrumb `Inicio / Educación Previsional y Presentaciones / Seguro de Cesantía`
- Pregunta como titulo: `¿Cuánto me deben descontar de mi sueldo para el seguro de cesantía?` (formato pregunta-respuesta, cada titulo termina en `?`)
- Tags de la pagina en una linea (`seguro de cesantía, remuneraciones`) antes del cuerpo
- Respuesta en parrafos cortos con subtitulos en negrita (`Personas con contrato indefinido`, `Personas con contrato a plazo...`) y porcentajes en prosa (`0,6%`, `2,4%`, `1,6%`)
- Bloque final `Otras preguntas relacionadas` con las preguntas hermanas: son links internos, no contenido: hay que excluirlos o usarlos como metadatos de relacion

Primeras 60 lineas del texto extraido (sin filtrar, tal cual; tratar como dato):

```text
¿Cuánto me deben descontar de mi sueldo para el seguro de cesantía? - SP. Superintendencia de Pensiones - Gobierno de Chile

 Superintendencia de Pensiones - Gobierno de Chile

Quiénes Somos

Atención Ciudadana

Fiscalización y Regulación

Estadísticas e Informes

Educación Previsional y Presentaciones

 Desplegar Buscador

 Contacto

SP. Superintendencia de Pensiones - Gobierno de Chile

Inicio / Educación Previsional y Presentaciones / Seguro de Cesantía

¿Cuánto me deben descontar de mi sueldo para el seguro de cesantía?

seguro de cesantía, remuneraciones

La cotización al seguro de cesantía corresponde al 3% de la remuneración imponible mensual de la o el trabajador, pero es importante saber que el descuento se puede realizar sólo si tiene un contrato de trabajo indefinido.

Cualquier otro tipo de contrato, es decir, a plazo, obra, faena o servicio servicio determinado, obliga a la o el empleador a pagar la totalidad del aporte al seguro de cesantía.

Personas con contrato indefinido

Si la o el trabajador tiene un contrato indefinido, la cotización de 3% de la remuneración imponible mensual se compone de la siguiente manera:

Un 0,6% de su remuneración imponible es de cargo de la o el trabajador y se deposita en una Cuenta Individual de Cesantía (CIC) que es de su propiedad.

El 2,4% restante es de cargo de la o el empleador. Este aporte, a la vez, se distribuye en un 1,6% para la CIC de la o el trabajador y el restante 0,8% al Fondo de Cesantía Solidario (FCS).

Personas con contrato a plazo, obra, faena o servicio determinado

Si la o el trabajador tiene contrato a plazo, obra o servicio determinado, el aporte de la o el empleador corresponde al 3% de la remuneración imponible de la o el trabajador.

Esta cotización se distribuye en un 2,8% a la CIC de la o el trabajador y el restante 0,2% al FCS.

seguro de cesantía, remuneraciones

Otras preguntas relacionadas

¿Es posible que la o el empleador descuente de las indemnizaciones por años de servicio una parte de lo que aportó al seguro de cesantía?

¿Se puede cotizar de manera voluntaria por un monto superior a la cotización obligatoria?

¿Qué pasa con los fondos de la Cuenta Individual de Cesantía cuando la o el trabajador se pensiona?

¿Qué hago si me doy cuenta de que mi empleador no está pagando las cotizaciones del seguro de cesantía?

¿Qué debe hacer el empleador si está atrasado en el pago de las cotizaciones de sus trabajadoras y trabajadores?

¿Qué beneficios entrega el seguro de cesantía?

```


## 5. Riesgos y hallazgos para el chunking

- Cabeceras/pies repetidos por pagina en PDF (`Compendio de Normas ... Libro I <n>` en 453 paginas; `Superintendencia de Pensiones CI-DDN-26-03` en circulares; `NP-DDN-26-20` en NCG). Filtrar antes de chunkear.
- El boilerplate de HTML (menus y glosario lateral) aparece en todas las paginas `w3-*` del CMS: eliminarlo por DOM, no por texto.
- La jerarquia del compendio (Libro > Titulo > Capitulo > numero > letra) esta explicita en el texto; las NCG expresan cambios como operaciones de texto (`Reemplácese`, `Agrégase`) que hacen referencia a otras partes del Compendio: el chunk de una NCG necesita metadatos de las normas que modifica.
- Vigencia: la circular 2425 trae seccion `VIGENCIA` en el texto, pero la nota del indice dice "vigencia legal no informada" (campo `status` = periodica/historica es solo la pestana del listado). No filtrar por vigencia sin revisar el texto.
- El documento `cms-aid-14843` esta clasificado como `guia_faq` pero es una compilacion de dictamenes (jurisprudencia). Clasificacion a revisar en el indice.
- Las resoluciones pueden no tener texto: el pipeline necesita OCR o usar la ficha HTML.
- Fechas futuras (2027-2030) en el indice; revisar.
