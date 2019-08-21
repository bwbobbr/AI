'''
案例名称 Instacart Market Basket Analysis
案例网站’https://www.kaggle.com/c/instacart-market-basket-analysis/data‘
文件 aisles.csv   aisle_id,aisle
文件 departments.csv  department_id,department
文件 order_products__*.csv   order_id,product_id,add_to_cart_order,reordered
文件 orders.csv   order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order
文件 products.csv    product_id,product_name,aisle_id,department_id
文件 sample_submission.csv  order_id,products\


数据量较大在jupyter上运行,位置为C:\Users\bob\数据降维案例
'''
import pandas as pd
from sklearn.decomposition import PCA

# 读取四张表的数据
prior = pd.read_csv("B:/PyCharm_Python_Project/PyCharm/AI/instance_ex/instacart-market-basket-analysis/order_products__prior.csv")
# 特征  order_id,product_id,add_to_cart_order,reordered

products = pd.read_csv('B:/PyCharm_Python_Project/PyCharm/AI/instance_ex/instacart-market-basket-analysis/products.csv')
# 特征 product_id,product_name,aisle_id,department_id

orders = pd.read_csv('B:/PyCharm_Python_Project/PyCharm/AI/instance_ex/instacart-market-basket-analysis/orders.csv')
# 特征文件 order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order

aisles = pd.read_csv('B:/PyCharm_Python_Project/PyCharm/AI/instance_ex/instacart-market-basket-analysis/aisles.csv')
# 特征 aisle_id,aisle

# 合并四张表到一张表(用户--物品类别)
_mg = pd.merge(prior,products,on=['product_id','product_id'])
_mg = pd.merge(_mg,orders,on=['order_id','order_id'])
mt = pd.merge(_mg,aisles,on=['aisle_id','aisle_id'])
print(mt.head(10))
# print(aisles)
















