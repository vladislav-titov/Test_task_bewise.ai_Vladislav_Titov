#!/usr/bin/env python
# coding: utf-8

# In[341]:


import pandas as pd
import re
from natasha import (Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger, PER, 
                     NamesExtractor, Doc)
import requests as rq
from bs4 import BeautifulSoup as BS
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.display.max_colwidth = None
data = pd.read_csv('D:/Загрузки/test_data.csv', sep = ',')
#1. Извлекать реплики с приветствием – где менеджер поздоровался. 
greeting = data[(data['role'] == 'manager') & (data['text'].str.contains("здравствуйте") 
                                               | data['text'].str.contains("Здравствуйте") 
                                               | data['text'].str.contains("Добрый день") 
                                               | data['text'].str.contains("добрый день"))]
#2. Извлекать реплики, где менеджер представил себя. 
presentation = data[(data['role'] == 'manager') & (data['text'].str.contains("зовут"))]
#2 или 3. Извлекать имя менеджера. 
#Получаем базу имен с сайта и подчищаем ее от дефектных элементов (Это сделано потому, что библиотека natasha 
#регистро- и пунктационно- чувствительная. Если бы текст был предварительно обработан, то можно было бы просто извлечь сущности
#из реплик (имена, организации, локации и другие имена собственные). Так как этого не сделано, значит остается два возможных вы-
#-хода:
#1) заимствовать или составить базу имен из какого-то источника и сверять слова из реплик с ней
#2) предварительно обработать текст на предмет синтаксиса и грамматики русского языка
#1 способ довольно быстрый, поэтому я решил пойти им. 2 способ довольно затратный по времени, поэтому я его не использую в
#условиях ограниченного времени.
r = rq.get('http://imja.name/imena/slovar-imen.shtml')
soup = BS(r.text, 'html.parser')
soup = soup.findAll('div', {'class': 'text'})[3].findAll('blockquote')[0]
for x, y in zip(soup.select('a'), soup.select('div')):
    x.decompose()
    y.decompose()
soup = re.sub('имен вместе с вариантами', '', soup.text)
soup = re.sub('\n', '', soup)
soup = re.sub('[0-9]+', '', soup)
soup = re.sub('\s-\s', '', soup)
soup = re.sub('»Имена на букву [А-Я]{1}', '', soup)
soup = re.sub('имя вместе с вариантами', '', soup)
soup = re.sub('имени вместе с вариантами', '', soup)
soup = re.sub('»им[а-я]+ на буквы [А-Я\,]{1,3}', '', soup)
soup = soup.replace(' – ', '').replace('(', '').replace(')', '').replace('\s+', ' ').strip().lower()
soup = soup.replace('»имена на буквы и, й', '')
soup = soup.replace(' ли ', ' ')
soup = soup.replace(' сами ', ' ')
#Замена возможных латинских символов на сходные русские
soup = soup.replace('a', 'а')
soup = soup.replace('A', 'А')
soup = soup.replace('B', 'В')
soup = soup.replace('c', 'с')
soup = soup.replace('C', 'С')
soup = soup.replace('e', 'е')
soup = soup.replace('E', 'Е')
soup = soup.replace('H', 'Н')
soup = soup.replace('K', 'К')
soup = soup.replace('M', 'М')
soup = soup.replace('o', 'о')
soup = soup.replace('O', 'О')
soup = soup.replace('p', 'р')
soup = soup.replace('P', 'Р')
soup = soup.replace('T', 'Т')
soup = soup.replace('X', 'Х')
soup = soup.replace('y', 'у')
names = soup.split(' ')
#Извлекаем имя менеджера. 
midterm_list = []
items = []
char = []
noun = []
word = []
final_names = []
segmenter = Segmenter()
morph_vocab = MorphVocab()
#part.pos = NOUN, part.feats ={'Animacy': 'Anim', 'Case': 'Nom', 'Gender': 'Fem', 'Number': 'Sing'}
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
names_extractor = NamesExtractor(morph_vocab)
    
for i in range(0, len(data)):
    text = data['text'][i].lower()
    #Замена возможных латинских символов, похожих на кириллические символы, на аналогичные кириллические
    text = text.replace('a', 'а')
    text = text.replace('c', 'с')
    text = text.replace('e', 'е')
    text = text.replace('o', 'о')
    text = text.replace('p', 'р')
    text = text.replace('y', 'у')
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    sent = doc.text.split(' ')
    for part in sent:
        if part in names:
            word.append(part)
        elif part not in names:
            pass
    if len(word) != 0:
        final_names.append(word[0].capitalize())
    else:
        final_names.append(False)
    word.clear()

data['manager_name'] = final_names
names_dict = {}
for i in data['dlg_id'].unique():
    a = data[(data['manager_name'] != False) & (data['role'] == 'manager') & (data['dlg_id'] == i)]
    a = a.reset_index()
    names_dict[i] = a['manager_name'][0]
for i in range(len(final_names)):
    if final_names[i] not in list(set(names_dict.values())):
        final_names[i] == False
data['manager_name'] = final_names
data['manager_name'] = data.manager_name.replace(names_dict)
#Фильтрация имен менеджеров через ключевое слово "зовут"
basis = data[(data['manager_name'] != False) & (data['role'] == 'manager') & (data['text'].str.contains('зовут'))]
#Фильтрация через номера строк диалога и отсутствия слова "Алло" со стороны менеджера
basis = data[(data['manager_name'] != False) & (data['role'] == 'manager') & (data['text'].str.contains("Алло") == False) & (data['line_n'] <= 3)]
basis
#4. Извлекать название компании.
midterm_list = []
company_names = []
for item in data[(data['text'].str.contains('компания') | data['text'].str.contains('Компания')) & (data['text'].str.contains('зовут'))]['text']:
    #Замена возможных латинских символов, похожих на кириллические символы, на аналогичные кириллические
    text = item.lower()
    text = text.replace('a', 'а')
    text = text.replace('c', 'с')
    text = text.replace('e', 'е')
    text = text.replace('o', 'о')
    text = text.replace('p', 'р')
    text = text.replace('y', 'у')
    ner_tagger = NewsNERTagger(emb)
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    sent = doc.sents[0].tokens
    start_point = int()
    end_point = int()
    for i in range(len(sent)):
        if sent[i].text != 'компания':
            continue
        else:
            start_point = i
            break
    for a in range(start_point + 1, len(sent)):
        if sent[a].pos == 'VERB':
            if sent[a].feats['Aspect'] == 'Imp':
                if sent[a].feats['Voice'] == 'Act':  #{'Aspect': 'Imp', 'Voice': 'Act'}:
                    end_point = a
                    break
        else:
            continue
    for i in range(start_point + 1, end_point):
        if sent[i].pos != 'ADJ':
            midterm_list.append(sent[i].text)
    if end_point - start_point == 1:
        midterm_list.append(sent[end_point].text)
    company_names.append(' '.join(midterm_list).capitalize())
    midterm_list.clear()
#print(start_point, end_point, company_names, sent)
company = {}
for i in range(len(company_names)):
    company[i] = company_names[i]
for n in data['dlg_id'].unique():
    if n in company.keys():
        pass
    else:
        company[n] = 'Компания не упоминалась в разговоре'
data['company'] = data.dlg_id.replace(company)
#5. Извлекать реплики, где менеджер попрощался.
aufwiedersien = data[(data['text'].str.contains('до свидания') | data['text'].str.contains('До свидания') | data['text'].str.contains('всего доброго')) & (data['role'] == 'manager')]
aufwiedersien
#6. Проверять требование к менеджеру: «В каждом диалоге обязательно необходимо поздороваться и попрощаться с клиентом»
check = {}
for i in greeting['dlg_id'].unique():
    if i in aufwiedersien['dlg_id'].unique():
        check[i] = 'Менеджер поздоровался и попрощался'
for n in data['dlg_id'].unique():
    if n in check.keys():
        pass
    else:
        check[n] = 'Менеджер не вежливый'
data['check_politeness'] = data.dlg_id.replace(check)
data

