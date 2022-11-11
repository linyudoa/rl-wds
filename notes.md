充实对RL模型的介绍 √
对比模型介绍：NM、PSO √
RL的优势介绍：训练快速、实时预测、运行时不需模型 √
介绍数据集，介绍生成手段 √
介绍结果，放图：训练过程、测试结果得分 √
介绍未来的实验计划: 使用GNN+RL/实时数据 + NM分别训练历史管网，探索泛化效果 

搞清楚其他pump如何工作，在模型中是否能作为当时的常量: Can be set according to history state  √
写一个parser处理.inp的pattern等数据，将basedemand + pattern数据处理成数据集: √

validate the model by:

1. Train and validate on real history data
2. Train on random data and validate on history data
3. print original pump op and predicted op
4. Calc history reward and compare it to validate reward
5. Compare head, energy efficiency of history and validate data
