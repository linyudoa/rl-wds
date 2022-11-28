充实对RL模型的介绍 √
对比模型介绍：NM、PSO √
RL的优势介绍：训练快速、实时预测、运行时不需模型 √
介绍数据集，介绍生成手段 √
介绍结果，放图：训练过程、测试结果得分 √
介绍未来的实验计划: 使用GNN+RL/实时数据 + NM分别训练历史管网，探索泛化效果 

搞清楚其他pump如何工作，在模型中是否能作为当时的常量: Can be set according to history state  √
写一个parser处理.inp的pattern等数据，将basedemand + pattern数据处理成数据集: √

HEAD DICT:
HUAYI: 华益
QINGDONG: 青东
J40543: 华新
JIAHUA: 嘉华
JIXI: 纪西
J49895: 纪王
J59970: 农科
J110064: 火星
J107956: 新凤
J79998: 白杨
HUAYU: 华宇
J111568: 诸新
J13356: 金联
J82123 : 平野 
XULE: 徐乐
J95051: 迮奄
JINSHUI: 金水
J56841: 广虹
PANZHONG: 蟠中
J77098: 国展
HUAQING: 华青
ZHUGUANG: 诸光
J54945: 邮政
J101196: 沪外
诸蟠 is not avalable

validate the model by:
1. Train and validate on real history data
2. Use water head dict
3. Train on random data and validate on history data
4. print original pump op and predicted op
5. Calc history reward and compare it to validate reward
6. Compare head, energy efficiency of history and validate data

Problems:
    Need to figure out how to use wdm model on epanet **done**
    Figure out what does tank/reservoir pattern mean **done**

need to replay complete scene by emitting details of temporal simulation **done**

1. replay and emit data like tank level of snapshots **done**
2. apply tank level while applying snapshots **done**
   
also need to use hiscorical water demand porpotion to init randomized demands **todo after December**

todo from 11.21:
1. Modify reward logic to mimic QD eval planning **DONE**
2. Investigate model accuracy **DONE**
3. Figure out how to calc pump energy consumption -- from speed to energy **DONE**
4. Train and validate, mainly focus on energy eff
5. Visualize Ops by saving them, add logic to plot

advans of the model: 
1. Can predict outcome without munipulating the real model

todo:
1. figure out historical score and ctr head value, if too weird, modify pump status
2. figure out what's happening when ctr head is too low, containing pump flow & head as well as reservoir head
3. Extend epynet to add logic to change pump speed