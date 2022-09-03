#!/usr/bin/env python
# coding: utf-8

# In[343]:


#импортируем библиотеки
import pandas as pd
import re
from natasha import (Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger, PER, 
                     NamesExtractor, Doc)
import requests as rq
from bs4 import BeautifulSoup as BS

#выставляем настройки для отображения датафрейма, чтобы было удобно просматривать.
pd.set_option('display.max_rows', None) #снятие ограничений на количество максимально отражаемых строк
pd.set_option('display.max_columns', None) #снятие ограничений на количество максимально отражаемых столбцов

#загружаем данные тестового задания. В принципе можно сделать в виде функции и интегрировать в графический интерфейс
#(например, сделать GUI в tkinter, PyQT5 и других подобных пакетах, и забиндить выполнение функции на кнопку, предварительно
#выбирая через .askopenfiles() файл на компьютере)
data = pd.read_csv('D:/Загрузки/test_data.csv', sep = ',')

#1. Извлекать реплики с приветствием – где менеджер поздоровался. 
#Просто фильтруем датафрейм по следующим параметрам: реплика должна принадлежать менеджеру, и он должен сказать слово
#"здравствуйте". В принципе, скорее всего, можно игнорировать регистр слова, пропуская реплики через цикл, но не успел реализо-
#-вать сейчас из-за нехватки времени. Отфильтровованный датафрейм помещаем в переменную greeting.
greeting = data[(data['role'] == 'manager') & (data['text'].str.contains("здравствуйте") 
                                               | data['text'].str.contains("Здравствуйте") 
                                               | data['text'].str.contains("Добрый день") 
                                               | data['text'].str.contains("добрый день"))]

#2. Извлекать реплики, где менеджер представил себя. 
#Снова фильтруем датафрейм: реплика должна принадлежать менеджеру, и он должен сказать слово
#"зовут". С регистром аналогично, как и в первом. Отфильтровованный датафрейм помещаем в переменную presentation.
presentation = data[(data['role'] == 'manager') & (data['text'].str.contains("зовут"))]

#3. Извлекать имя менеджера. 
#Получаем базу имен с сайта imja.name и подчищаем ее от дефектных элементов (Это сделано потому, что библиотека natasha 
#регистро- и пунктационно- чувствительная. Если бы реплики были предварительно обработаны, то можно было бы просто извлечь сущ-
#-ности из реплик (имена, организации, локации и другие имена собственные). Так как этого не сделано, значит остается два возмо-
#-жных выхода:
#1) заимствовать или составить базу имен из какого-то источника и сверять слова из реплик с ней
#2) предварительно обработать текст на предмет синтаксиса и грамматики русского языка
#1 способ довольно быстрый, поэтому я решил пойти им. 2 способ довольно затратный по времени, поэтому я его не использую в
#условиях ограниченного времени.
r = rq.get('http://imja.name/imena/slovar-imen.shtml') #соединяемся с сайтом через requests
soup = BS(r.text, 'html.parser') #будем парсить через BeautifulSoup
#Выбираем нужную нам информацию (имена с сайта). Для этого по максимуму чистим лишние теги, все, до которых можем добраться.
soup = soup.findAll('div', {'class': 'text'})[3].findAll('blockquote')[0]
for x, y in zip(soup.select('a'), soup.select('div')):
    x.decompose()
    y.decompose()
#Преобразуем подчищенный bs4-объект в текст и очищаем до конца от лишнего с помощью функций регулярных выражений  
soup = re.sub('имен вместе с вариантами', '', soup.text) #Очищаем от заголовков на сайте
soup = re.sub('\n', '', soup) #Удаляем отступы абзаца
soup = re.sub('[0-9]+', '', soup) #Удаляем цифры
soup = re.sub('\s-\s', '', soup) #Удаляем тире с пробелами перед и после низ
soup = re.sub('»Имена на букву [А-Я]{1}', '', soup) #Очищаем от заголовков на сайте
soup = re.sub('имя вместе с вариантами', '', soup) #Очищаем от заголовков на сайте
soup = re.sub('имени вместе с вариантами', '', soup) #Очищаем от заголовков на сайте
soup = re.sub('»им[а-я]+ на буквы [А-Я\,]{1,3}', '', soup) #Очищаем от заголовков на сайте
soup = soup.replace(' – ', '').replace('(', '').replace(')', '').replace('\s+', ' ').strip().lower() #Удаляем тире, скобки, и
#заменяем много последовательных пробелов на одинарные пробел, и удаляем пробелы в начале и конце.  
soup = soup.replace('»имена на буквы и, й', '') #Чистим
soup = soup.replace(' ли ', ' ') #Чистим
soup = soup.replace(' сами ', ' ') #Чистим
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
#После очистки от лишнего и замены латинских символов на кириллические создаем список имен, деля текст по пробелам с помощью
#функции .split(). Список упаковываем в переменную names.
names = soup.split(' ')
#Создаем промежуточные пустые списки и списки, которые будем помещать в дальнейшем в датафрейм в качестве содержимого колонок.
items = []
word = []
final_names = []
#Ищем имена в репликах с помощью цикла и функций библиотеки natasha.
for i in range(0, len(data)): #Проходимся по циклу по индексам строк датафрейма
    text = data['text'][i].lower() #Берем реплики из датафрейма с помощью цикла
    #Замена возможных латинских символов, похожих на кириллические символы, на аналогичные кириллические
    text = text.replace('a', 'а')
    text = text.replace('c', 'с')
    text = text.replace('e', 'е')
    text = text.replace('o', 'о')
    text = text.replace('p', 'р')
    text = text.replace('y', 'у')
    doc = Doc(text) #Создаем объект Doc() библиотеки natasha для обработки реплик
    doc.segment(segmenter) 
    doc.tag_morph(morph_tagger) #конструкция для осуществления морфологического анализа
    doc.parse_syntax(syntax_parser) #конструкция для осуществления синтаксического анализа
    sent = doc.text.split(' ') #Делим предложение на отдельные слова
    for part in sent: #пускаем цикл по словам в отдельных репликах
        #Если имя в списке с именами, который я извлек с сайта imja.name, то добавляем слово в промежуточный список word
        #Если же нет, то пропускаем слово
        if part in names: 
            word.append(part) 
        elif part not in names: 
            pass 
    #Если в список word были добавлены слова (имена), то берем первое имя из этого списка и добавляем в окончательный список
    #final_names, предварительно делая заглавную букву у имени с помощью функции .capitalize(). Если имен в списке не оказалось,
    #то добавляем в final_names ЛОЖЬ.
    if len(word) != 0: #
        final_names.append(word[0].capitalize()) 
    else: 
        final_names.append(False) 
    word.clear() # очищаем промежуточный список word, чтобы слова из старых итераций не оказывались при новых проходках по циклу

data['manager_name'] = final_names #Создаем колонку "manager_names" в датафрейме, заполняя её данными из списка final_names.
names_dict = {} #Создаем словарь, для окончательного определения имен менеджеров, в зависимости от номеров диалогов
for i in data['dlg_id'].unique(): #Проходимся в циклу с уникальными номерами диалогов
    #Фильтруем датафрейм по следующим параметрам: наличие имени менеджера, реплика принадлежит менеджеру, и реплики относятся
    #к одному диалогу (задаем номера диалогов проходками по циклу)
    a = data[(data['manager_name'] != False) & (data['role'] == 'manager') & (data['dlg_id'] == i)]
    #Сбрасываем индекс промежуточного датафрейма
    a = a.reset_index()
    #Вставляем в словарь names_dict записи, где в качестве ключа выступает номер диалога, а в качестве значения первое попавшееся
    #имя менеджера
    names_dict[i] = a['manager_name'][0]
for i in range(len(final_names)): #проходимся по элементам final_names циклом
    if final_names[i] not in list(set(names_dict.values())): #Если имя в final_names не находится в значениях ключей names_dict,
    #то заменяем значение на False
        final_names[i] == False #
data['manager_name'] = final_names #Обновляем значения датафрейма новым списком
data['manager_name'] = data.manager_name.replace(names_dict) #Заполняем пустые значения в датафрейме именами менеджеров

#4. Извлекать название компании.
midterm_list = [] #создаем промежуточный список
company_names = [] #создаем список, где буду названия компаний
#Проходимся циклом по репликам, где содержится слова "компания" и "зовут"
for item in data[(data['text'].str.contains('компания') | data['text'].str.contains('Компания')) & (data['text'].str.contains('зовут'))]['text']:
    #Замена возможных латинских символов, похожих на кириллические символы, на аналогичные кириллические
    text = item.lower() #делаем у всех букв нижний регистр
    text = text.replace('a', 'а')
    text = text.replace('c', 'с')
    text = text.replace('e', 'е')
    text = text.replace('o', 'о')
    text = text.replace('p', 'р')
    text = text.replace('y', 'у')
    #Создаем объекты библиотеки natasha для морфологического и синтаксического анализов.
    ner_tagger = NewsNERTagger(emb) 
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    sent = doc.sents[0].tokens #слова из реплик с их морфологическими характеристиками
    #Создаем целочисленные переменные start_point и end_point, которые будут выступать границами для извлечения названия компаний 
    #из реплик
    start_point = int()
    end_point = int()
    for i in range(len(sent)): #Проходимся циклом по словам в репликах
        #Если слово не является словом "компания", то пропускаем его, если является, то берем его индекс в реплике и выходим из
        #цикла
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
#Не успел дописать комментарии. До дедлайна шесть минут. Могу подробнее объяснить код на собеседовании.

