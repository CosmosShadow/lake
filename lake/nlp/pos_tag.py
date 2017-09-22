# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# 相关参考
# 中文词性 http://repository.upenn.edu/cgi/viewcontent.cgi?article=1039&context=ircs_reports
# TODO: http://blog.csdn.net/eli00001/article/details/75088444

tag_names = dict([
('DT', '确定词(这,那,该,每)'),
('CD', '概数词(好些,半,许多)'),
('VD', '序列词(第十)'),
('JJ', '介词(Adjective)'),
('JJR', 'Adjective, comparative'),
('JJS', 'Adjective, superlative'),
('''NN''', u'名词'),
('''NNP''', u'专用名词的单数形式'),
('''NNPS''', u'专用名词的复数形式'),
('''PDT''', u'前置限定词'),
('''POS''', u'所有格结束符'),
('''PRP''', u'人称代词'),
('''PRP$''', u'所有格代词'),
('''RB''', u'副词'),
('''RBR''', u'相对副词'),
('''RBS''', u'最高级副词'),
('''RP''', u'小品词'),
('''SYM''', u'符号（数学符号或特殊符号）|'),
('''TO''', u'To'),
('''UH''', u'叹词'),
('''VA''', u'动词'),
('''VC''', u'动词'),
('''VE''', u'动词'),
('''VV''', u'动词'),
('''VB''', u'动词的基本形式'),
('''VBD''', u'动词的过去式'),
('''VBG''', u'动词的动名词用法'),
('''VBN''', u'动词的过去分词'),
('''WP''', u'Wh-代词'),
('''WP$''', u'所有格wh-代词'),
('''WRB''', u'Wh-副词'),
('''#''', u'井号符'),
('''$''', u'美元符'),
('''.''', u'句号'),
(''',''', u'逗号'),
(''':''', u'分号，分隔符'),
('''(''', u'左括号'),
('''),''', u'右括号'),
('''"''', u'直双引号'),
("'", u'左单引号'),
('''"''', u'左双引号'),
("'", u'右单引号'),
('''"''', u'右双引号')])