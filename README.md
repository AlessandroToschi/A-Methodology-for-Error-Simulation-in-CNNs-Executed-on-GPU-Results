- [Cardinalities](#cardinalities)
  - [Add](#add)
  - [Batch Norm](#batch-norm)
  - [Biasadd](#biasadd)
  - [Convolution 1](#convolution-1)
  - [Convolution 2](#convolution-2)
  - [Convolution 3](#convolution-3)
  - [Div](#div)
  - [Exp](#exp)
  - [Leaky ReLU](#leaky-relu)
  - [Mul](#mul)
  - [Sigmoid](#sigmoid)

# Cardinalities

## Add

|Cardinality|IOV|IOA|RF||Probability|
|:---:|---:|---:|---:|---:|---:|
|*1*|2787|2242|299|5328|*90.2744%*|
|*2*|91|390|66|547|*9.268%*|
|*3*|0|27|0|27|*0.4574%*|
|||||||
||*2878*|*2659*|*365*|*5902*|*100%*|

## Batch Norm

|Cardinality|IOV|IOA|RF||Probability|
|:---:|---:|---:|---:|---:|---:|
|*1*|13170|5623|1566|20359|*77.7595%*|
|*2*|562|1337|174|2073|*7.9176%*|
|*3*|13|3|0|16|*< 1‰*|
|*4*|15|3|0|18|*< 1‰*|
|*5*|14|3|0|17|*< 1‰*|
|*6*|11|7|0|18|*< 1‰*|
|*7*|12|3|1|16|*< 1‰*|
|*8*|10|8|1|19|*< 1‰*|
|*9*|22|18|3|43|*0.1642%*|
|*10*|9|6|0|15|*< 1‰*|
|*11*|17|5|1|23|*< 1‰*|
|*12*|21|6|0|27|*0.1031%*|
|*13*|36|6|0|42|*0.1604%*|
|*14*|30|18|1|49|*0.1871%*|
|*15*|156|93|15|264|*1.0083%*|
|*16*|1621|958|77|2656|*10.1443%*|
|*17*|9|26|1|36|*0.1374%*|
|*22*|1|0|0|1|*< 1‰*|
|*23*|0|2|0|2|*< 1‰*|
|*24*|3|6|1|10|*< 1‰*|
|*25*|1|0|0|1|*< 1‰*|
|*26*|0|2|0|2|*< 1‰*|
|*27*|0|1|0|1|*< 1‰*|
|*28*|1|1|0|2|*< 1‰*|
|*29*|1|5|0|6|*< 1‰*|
|*30*|2|19|0|21|*< 1‰*|
|*31*|27|78|1|106|*0.4048%*|
|*32*|6|62|1|69|*0.2635%*|
|*33*|1|1|0|2|*< 1‰*|
|*46*|0|2|0|2|*< 1‰*|
|*53*|0|2|0|2|*< 1‰*|
|*56*|0|1|0|1|*< 1‰*|
|*71*|0|1|0|1|*< 1‰*|
|*110*|1|0|0|1|*< 1‰*|
|*112*|0|1|0|1|*< 1‰*|
|*132*|1|0|0|1|*< 1‰*|
|*135*|1|0|0|1|*< 1‰*|
|*141*|1|0|0|1|*< 1‰*|
|*144*|0|1|0|1|*< 1‰*|
|*146*|1|0|0|1|*< 1‰*|
|*156*|1|0|0|1|*< 1‰*|
|*157*|0|1|0|1|*< 1‰*|
|*161*|0|1|0|1|*< 1‰*|
|*166*|2|0|0|2|*< 1‰*|
|*167*|1|0|0|1|*< 1‰*|
|*168*|3|3|0|6|*< 1‰*|
|*169*|96|111|1|208|*0.7944%*|
|*170*|1|2|0|3|*< 1‰*|
|*210*|0|1|0|1|*< 1‰*|
|*227*|0|1|0|1|*< 1‰*|
|*232*|0|1|0|1|*< 1‰*|
|*256*|2|0|0|2|*< 1‰*|
|*264*|0|1|0|1|*< 1‰*|
|*306*|0|1|0|1|*< 1‰*|
|*311*|0|1|0|1|*< 1‰*|
|*332*|0|1|0|1|*< 1‰*|
|*337*|0|2|0|2|*< 1‰*|
|*338*|1|8|1|10|*< 1‰*|
|*413*|0|1|0|1|*< 1‰*|
|*519*|0|1|0|1|*< 1‰*|
|*665*|0|1|0|1|*< 1‰*|
|*1221*|0|1|0|1|*< 1‰*|
|*1241*|0|1|0|1|*< 1‰*|
|*2704*|2|3|1|6|*< 1‰*|
|||||||
||*15885*|*8451*|*1846*|*26182*|*100%*|

## Biasadd

|Cardinality|IOV|IOA|RF||Probability|
|:---:|---:|---:|---:|---:|---:|
|*1*|3965|2361|412|6738|*90.0922%*|
|*2*|120|543|77|740|*9.8943%*|
|*3*|1|0|0|1|*< 1‰*|
|||||||
||*4086*|*2904*|*489*|*7479*|*100%*|

## Convolution 1

|Cardinality|IOV|IOA|RF||Probability|
|:---:|---:|---:|---:|---:|---:|
|*1*|3788|5353|1220|10361|*42.6817%*|
|*2*|229|91|11|331|*1.3635%*|
|*3*|192|62|8|262|*1.0793%*|
|*4*|207|69|13|289|*1.1905%*|
|*5*|216|78|11|305|*1.2564%*|
|*6*|247|79|10|336|*1.3841%*|
|*7*|259|80|15|354|*1.4582%*|
|*8*|257|109|31|397|*1.6354%*|
|*9*|434|132|117|683|*2.8135%*|
|*10*|307|112|31|450|*1.8537%*|
|*11*|316|138|41|495|*2.0391%*|
|*12*|359|144|40|543|*2.2368%*|
|*13*|475|202|56|733|*3.0195%*|
|*14*|533|293|101|927|*3.8187%*|
|*15*|679|379|361|1419|*5.8455%*|
|*16*|947|495|1040|2482|*10.2245%*|
|*17*|16|13|48|77|*0.3171%*|
|*18*|14|7|19|40|*0.1647%*|
|*19*|19|14|7|40|*0.1647%*|
|*20*|17|12|4|33|*0.1359%*|
|*21*|20|22|10|52|*0.2142%*|
|*22*|32|19|8|59|*0.243%*|
|*23*|34|26|19|79|*0.3254%*|
|*24*|111|22|34|167|*0.6879%*|
|*25*|28|18|5|51|*0.21%*|
|*26*|31|19|12|62|*0.2554%*|
|*27*|54|36|21|111|*0.4572%*|
|*28*|80|23|42|145|*0.5973%*|
|*29*|191|37|85|313|*1.2893%*|
|*30*|517|85|216|818|*3.3697%*|
|*31*|1183|111|454|1748|*7.2008%*|
|*32*|7|14|83|104|*0.4284%*|
|*40*|0|1|0|1|*< 1‰*|
|*43*|0|1|0|1|*< 1‰*|
|*46*|0|4|1|5|*< 1‰*|
|||||||
||*11799*|*8300*|*4174*|*24273*|*100%*|

## Convolution 2

|Cardinality|IOV|IOA|RF||Probability|
|:---:|---:|---:|---:|---:|---:|
|*1*|1749|3744|849|6342|*40.9478%*|
|*2*|115|122|14|251|*1.6206%*|
|*3*|95|72|20|187|*1.2073%*|
|*4*|106|62|16|184|*1.188%*|
|*5*|124|66|9|199|*1.2848%*|
|*6*|112|79|16|207|*1.3365%*|
|*7*|102|76|16|194|*1.2525%*|
|*8*|132|90|19|241|*1.556%*|
|*9*|131|90|16|237|*1.5302%*|
|*10*|140|80|30|250|*1.6141%*|
|*11*|135|133|31|299|*1.9305%*|
|*12*|159|127|50|336|*2.1694%*|
|*13*|205|166|72|443|*2.8602%*|
|*14*|247|217|129|593|*3.8287%*|
|*15*|275|258|300|833|*5.3783%*|
|*16*|407|296|588|1291|*8.3354%*|
|*17*|19|20|49|88|*0.5681%*|
|*18*|9|9|7|25|*0.1614%*|
|*19*|16|13|4|33|*0.213%*|
|*20*|12|17|8|37|*0.2388%*|
|*21*|19|18|6|43|*0.2776%*|
|*22*|19|16|6|41|*0.2647%*|
|*23*|24|14|11|49|*0.3163%*|
|*24*|21|9|9|39|*0.2518%*|
|*25*|27|20|6|53|*0.3422%*|
|*26*|26|18|25|69|*0.4455%*|
|*27*|43|25|45|113|*0.7295%*|
|*28*|35|30|43|108|*0.6973%*|
|*29*|102|41|102|245|*1.5818%*|
|*30*|127|70|153|350|*2.2598%*|
|*31*|187|46|192|425|*2.744%*|
|*32*|10|9|34|53|*0.3422%*|
|*33*|6|4|1|11|*< 1‰*|
|*34*|7|2|1|10|*< 1‰*|
|*35*|6|3|1|10|*< 1‰*|
|*36*|6|4|2|12|*< 1‰*|
|*37*|14|7|1|22|*0.142%*|
|*38*|10|5|1|16|*0.1033%*|
|*39*|7|6|0|13|*< 1‰*|
|*40*|6|5|4|15|*< 1‰*|
|*41*|6|6|2|14|*< 1‰*|
|*42*|9|10|2|21|*0.1355%*|
|*43*|8|5|2|15|*< 1‰*|
|*44*|11|10|2|23|*0.1485%*|
|*45*|5|7|3|15|*< 1‰*|
|*46*|12|7|4|23|*0.1485%*|
|*47*|13|9|0|22|*0.142%*|
|*48*|20|9|1|30|*0.1936%*|
|*49*|13|7|4|24|*0.1549%*|
|*50*|23|7|2|32|*0.2066%*|
|*51*|20|5|8|33|*0.213%*|
|*52*|26|10|5|41|*0.2647%*|
|*53*|26|13|5|44|*0.284%*|
|*54*|20|9|3|32|*0.2066%*|
|*55*|23|18|13|54|*0.3486%*|
|*56*|29|23|9|61|*0.3938%*|
|*57*|41|20|9|70|*0.4519%*|
|*58*|38|13|3|54|*0.3486%*|
|*59*|41|24|14|79|*0.51%*|
|*60*|53|32|12|97|*0.6262%*|
|*61*|69|41|13|123|*0.7941%*|
|*62*|86|36|28|150|*0.9684%*|
|*63*|85|40|24|149|*0.962%*|
|*64*|65|29|25|119|*0.7683%*|
|*68*|0|1|0|1|*< 1‰*|
|*71*|0|2|0|2|*< 1‰*|
|*72*|1|1|0|2|*< 1‰*|
|*73*|0|1|0|1|*< 1‰*|
|*74*|0|1|0|1|*< 1‰*|
|*75*|0|1|0|1|*< 1‰*|
|*80*|0|2|0|2|*< 1‰*|
|*81*|0|1|0|1|*< 1‰*|
|*83*|1|0|0|1|*< 1‰*|
|*85*|0|1|1|2|*< 1‰*|
|*87*|0|1|0|1|*< 1‰*|
|*89*|1|1|0|2|*< 1‰*|
|*90*|1|0|1|2|*< 1‰*|
|*93*|0|1|0|1|*< 1‰*|
|*95*|0|2|0|2|*< 1‰*|
|*96*|1|0|0|1|*< 1‰*|
|*97*|0|1|0|1|*< 1‰*|
|*98*|0|0|1|1|*< 1‰*|
|*99*|1|0|0|1|*< 1‰*|
|*100*|1|2|0|3|*< 1‰*|
|*101*|0|3|0|3|*< 1‰*|
|*102*|2|1|0|3|*< 1‰*|
|*103*|0|1|0|1|*< 1‰*|
|*104*|1|0|0|1|*< 1‰*|
|*105*|1|2|0|3|*< 1‰*|
|*106*|1|0|1|2|*< 1‰*|
|*107*|0|0|1|1|*< 1‰*|
|*109*|3|1|0|4|*< 1‰*|
|*110*|1|3|0|4|*< 1‰*|
|*111*|1|0|2|3|*< 1‰*|
|*112*|1|2|1|4|*< 1‰*|
|*113*|2|4|0|6|*< 1‰*|
|*114*|2|2|2|6|*< 1‰*|
|*115*|2|2|1|5|*< 1‰*|
|*116*|4|2|2|8|*< 1‰*|
|*117*|5|0|2|7|*< 1‰*|
|*118*|8|1|0|9|*< 1‰*|
|*119*|3|0|1|4|*< 1‰*|
|*120*|6|5|2|13|*< 1‰*|
|*121*|1|1|3|5|*< 1‰*|
|*122*|5|6|1|12|*< 1‰*|
|*123*|7|2|2|11|*< 1‰*|
|*124*|6|2|3|11|*< 1‰*|
|*125*|5|3|2|10|*< 1‰*|
|*126*|7|2|3|12|*< 1‰*|
|*127*|5|7|1|13|*< 1‰*|
|*128*|3|1|2|6|*< 1‰*|
|||||||
||*5823*|*6551*|*3114*|*15488*|*100%*|

## Convolution 3

|Cardinality|IOV|IOA|RF||Probability|
|:---:|---:|---:|---:|---:|---:|
|*1*|4260|9422|2111|15793|*50.5456%*|
|*2*|48|109|9|166|*0.5312%*|
|*3*|39|18|8|65|*0.208%*|
|*4*|53|35|64|152|*0.4864%*|
|*5*|41|15|4|60|*0.192%*|
|*6*|27|14|4|45|*0.144%*|
|*7*|27|18|2|47|*0.1504%*|
|*8*|35|27|10|72|*0.2304%*|
|*9*|31|33|7|71|*0.2272%*|
|*10*|54|43|9|106|*0.3392%*|
|*11*|77|57|8|142|*0.4544%*|
|*12*|107|94|20|221|*0.7073%*|
|*13*|215|180|28|423|*1.3538%*|
|*14*|382|362|90|834|*2.6692%*|
|*15*|746|845|385|1976|*6.3242%*|
|*16*|1507|1700|1878|5085|*16.2746%*|
|*17*|6|7|101|114|*0.3648%*|
|*18*|4|10|1|15|*< 1‰*|
|*19*|10|6|5|21|*< 1‰*|
|*20*|5|3|0|8|*< 1‰*|
|*21*|5|5|0|10|*< 1‰*|
|*22*|7|2|0|9|*< 1‰*|
|*23*|4|10|1|15|*< 1‰*|
|*24*|4|7|3|14|*< 1‰*|
|*25*|5|13|1|19|*< 1‰*|
|*26*|13|27|7|47|*0.1504%*|
|*27*|12|27|4|43|*0.1376%*|
|*28*|23|51|17|91|*0.2912%*|
|*29*|61|74|46|181|*0.5792%*|
|*30*|174|153|198|525|*1.6802%*|
|*31*|607|335|845|1787|*5.7193%*|
|*32*|10|74|216|300|*0.9601%*|
|*33*|1|1|0|2|*< 1‰*|
|*34*|3|0|0|3|*< 1‰*|
|*35*|4|1|0|5|*< 1‰*|
|*36*|2|1|0|3|*< 1‰*|
|*37*|0|1|0|1|*< 1‰*|
|*38*|2|1|1|4|*< 1‰*|
|*39*|2|0|0|2|*< 1‰*|
|*40*|1|0|2|3|*< 1‰*|
|*41*|1|3|1|5|*< 1‰*|
|*42*|0|1|0|1|*< 1‰*|
|*43*|2|1|0|3|*< 1‰*|
|*44*|3|1|0|4|*< 1‰*|
|*45*|1|6|1|8|*< 1‰*|
|*46*|2|23|0|25|*< 1‰*|
|*47*|5|4|1|10|*< 1‰*|
|*48*|8|1|0|9|*< 1‰*|
|*49*|4|3|0|7|*< 1‰*|
|*50*|7|1|1|9|*< 1‰*|
|*51*|2|3|2|7|*< 1‰*|
|*52*|8|4|1|13|*< 1‰*|
|*53*|14|5|1|20|*< 1‰*|
|*54*|11|4|3|18|*< 1‰*|
|*55*|20|8|4|32|*0.1024%*|
|*56*|21|9|0|30|*< 1‰*|
|*57*|14|20|5|39|*0.1248%*|
|*58*|45|14|6|65|*0.208%*|
|*59*|41|32|10|83|*0.2656%*|
|*60*|56|38|21|115|*0.368%*|
|*61*|88|64|31|183|*0.5856%*|
|*62*|162|129|58|349|*1.1169%*|
|*63*|266|177|123|566|*1.8114%*|
|*64*|365|246|197|808|*2.586%*|
|*75*|0|1|0|1|*< 1‰*|
|*78*|0|1|0|1|*< 1‰*|
|*79*|0|1|0|1|*< 1‰*|
|*83*|0|2|0|2|*< 1‰*|
|*85*|1|0|0|1|*< 1‰*|
|*88*|0|1|0|1|*< 1‰*|
|*92*|0|1|0|1|*< 1‰*|
|*98*|0|1|0|1|*< 1‰*|
|*99*|0|1|0|1|*< 1‰*|
|*100*|0|2|0|2|*< 1‰*|
|*101*|0|2|0|2|*< 1‰*|
|*102*|0|1|0|1|*< 1‰*|
|*103*|0|1|0|1|*< 1‰*|
|*107*|0|1|0|1|*< 1‰*|
|*108*|0|1|0|1|*< 1‰*|
|*109*|0|1|0|1|*< 1‰*|
|*110*|0|2|0|2|*< 1‰*|
|*111*|0|1|0|1|*< 1‰*|
|*112*|1|1|0|2|*< 1‰*|
|*114*|1|1|0|2|*< 1‰*|
|*115*|0|2|1|3|*< 1‰*|
|*116*|2|2|0|4|*< 1‰*|
|*117*|0|2|0|2|*< 1‰*|
|*118*|1|5|0|6|*< 1‰*|
|*119*|1|2|0|3|*< 1‰*|
|*120*|1|2|1|4|*< 1‰*|
|*121*|3|1|0|4|*< 1‰*|
|*122*|1|6|0|7|*< 1‰*|
|*123*|6|7|0|13|*< 1‰*|
|*124*|12|13|3|28|*< 1‰*|
|*125*|11|22|7|40|*0.128%*|
|*126*|22|22|10|54|*0.1728%*|
|*127*|43|33|10|86|*0.2752%*|
|*128*|34|34|7|75|*0.24%*|
|*684*|0|1|0|1|*< 1‰*|
|||||||
||*9900*|*14755*|*6590*|*31245*|*100%*|

## Div

|Cardinality|IOV|IOA|RF||Probability|
|:---:|---:|---:|---:|---:|---:|
|*1*|2226|1343|206|3775|*84.9077%*|
|*2*|94|511|65|670|*15.0697%*|
|*4*|1|0|0|1|*< 1‰*|
|||||||
||*2321*|*1854*|*271*|*4446*|*100%*|

## Exp

|Cardinality|IOV|IOA|RF||Probability|
|:---:|---:|---:|---:|---:|---:|
|*1*|3212|2243|472|5927|*91.9057%*|
|*2*|49|349|124|522|*8.0942%*|
|||||||
||*3261*|*2592*|*596*|*6449*|*100%*|

## Leaky ReLU

|Cardinality|IOV|IOA|RF||Probability|
|:---:|---:|---:|---:|---:|---:|
|*1*|2374|1801|201|4376|*84.4787%*|
|*2*|193|532|77|802|*15.4826%*|
|*3*|1|0|1|2|*< 1‰*|
|||||||
||*2568*|*2333*|*279*|*5180*|*100%*|

## Mul

|Cardinality|IOV|IOA|RF||Probability|
|:---:|---:|---:|---:|---:|---:|
|*1*|2861|1934|319|5114|*88.4316%*|
|*2*|84|524|61|669|*11.5683%*|
|||||||
||*2945*|*2458*|*380*|*5783*|*100%*|

## Sigmoid

|Cardinality|IOV|IOA|RF||Probability|
|:---:|---:|---:|---:|---:|---:|
|*1*|1736|2087|283|4106|*89.7486%*|
|*2*|88|318|63|469|*10.2513%*|
|||||||
||*1824*|*2405*|*346*|*4575*|*100%*|