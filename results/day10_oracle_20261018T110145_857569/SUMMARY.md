# Oracle MPC固定网格结果

相同random shooting，只替换评分动力学；learned固定P0-1最佳Adam模型。

|seed|模式|N|H|真实回报|最长保持步数|成功|末20步稳定|
|---:|---|---:|---:|---:|---:|---|---|
|100|random|None|None|-1327.140389|0|False|False|
|100|learned|20|5|-1288.398146|0|False|False|
|100|oracle|20|5|-1218.098582|0|False|False|
|100|learned|200|5|-1258.619920|0|False|False|
|100|oracle|200|5|-1138.559167|0|False|False|
|100|learned|20|15|-364.490757|101|True|True|
|100|oracle|20|15|-240.770473|142|True|True|
|100|learned|200|15|-239.101602|149|True|True|
|100|oracle|200|15|-238.131855|153|True|True|
|100|learned|20|30|-603.778877|93|True|True|
|100|oracle|20|30|-601.217819|94|True|True|
|100|learned|200|30|-243.221166|154|True|True|
|100|oracle|200|30|-243.351960|153|True|True|
|101|random|None|None|-1542.926696|0|False|False|
|101|learned|20|5|-1516.859574|0|False|False|
|101|oracle|20|5|-1616.765690|0|False|False|
|101|learned|200|5|-1553.310790|0|False|False|
|101|oracle|200|5|-1553.346392|0|False|False|
|101|learned|20|15|-501.134801|115|True|True|
|101|oracle|20|15|-504.471940|110|True|True|
|101|learned|200|15|-392.241657|132|True|True|
|101|oracle|200|15|-378.950145|135|True|True|
|101|learned|20|30|-779.677337|80|True|True|
|101|oracle|20|30|-637.736184|102|True|True|
|101|learned|200|30|-492.739816|118|True|True|
|101|oracle|200|30|-485.857868|120|True|True|
|102|random|None|None|-1457.908154|0|False|False|
|102|learned|20|5|-1288.548639|0|False|False|
|102|oracle|20|5|-1221.820760|0|False|False|
|102|learned|200|5|-1264.594665|0|False|False|
|102|oracle|200|5|-1148.836929|0|False|False|
|102|learned|20|15|-246.415696|152|True|True|
|102|oracle|20|15|-245.429357|154|True|True|
|102|learned|200|15|-246.086569|151|True|True|
|102|oracle|200|15|-243.016460|154|True|True|
|102|learned|20|30|-369.854255|76|True|False|
|102|oracle|20|30|-243.336185|152|True|True|
|102|learned|200|30|-242.167713|155|True|True|
|102|oracle|200|30|-364.703389|132|True|True|

判读规则见运行前config.json；保持成功按执行后的真实状态计算，不把经过直立算保持。
三开发种子、单采样种子；oracle使用精确内部状态是诊断特权。失败结果全部保留。
未训练、未评估test、原来源哈希不变；完整轨迹/候选指纹/耗时见各episodes文件。
