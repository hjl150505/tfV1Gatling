"""
categorical_column_with_hash_bucket的使用
输入一维特征列
输入二维特征列
"""

import tensorflow as tf
# 该语句在tf2.1.0环境运行时必须添加，否则会报错raise RuntimeError('The Session graph is empty. Add operations to the '
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.Session()
# 特征数据
# features = {
#     'department': ['sport', 'sport', 'drawing', 'gardening', 'travelling'],
# }
features = {
    'department': [['sport', 'sport', 'drawing'], ['gardening', 'travelling', 'drawing']],
}
# 特征列
department = tf.feature_column.categorical_column_with_hash_bucket('department', 5, dtype=tf.string)
# department = tf.feature_column.indicator_column(department)
department = tf.feature_column.embedding_column(department, dimension=5, combiner='sum')
# 组合特征列
columns = [department]
# 输入层（数据，特征列）
inputs = tf.compat.v1.feature_column.input_layer(features, columns)
# 初始化并运行
init = tf.compat.v1.global_variables_initializer()
sess.run(tf.compat.v1.tables_initializer())
sess.run(init)

v = sess.run(inputs)
print(v)