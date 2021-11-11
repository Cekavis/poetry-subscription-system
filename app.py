from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from sklearn.decomposition import PCA
from threading import Thread
import webbrowser
import re
import numpy as np
import math
import os
import pickle

# Parse poems from file
def get_poems():
    poems = {}
    with open('poems.txt', 'r') as f:
        poem = {'content': []}
        for l in f.readlines():
            l = l.split(' ')
            line = re.sub(r'\([^)]*\)', '', ''.join(l[3:]))
            line = line.strip()
            if l[2]=='-100':
                if poem['content']:
                    poems[poem['author']] = poems.get(poem['author'], [])
                    poems[poem['author']].append(poem)
                poem = {'title': line, 'content': []}
            elif l[2]=='-10':
                assert 'title' in poem
                poem['note'] = line
            elif l[2]=='-1':
                assert 'title' in poem
                poem['author'] = line
            elif int(l[2])>0:
                assert 'title' in poem
                poem['content'].append(line)
                assert len(poem['content']) == int(l[2])

        if poem['content']:
            poems[poem['author']] = poems.get(poem['author'], [])
            poems[poem['author']].append(poem)
    return poems

poems = get_poems()
# poems['all'] stores all poems as a list
poems['all'] = [x for y in poems.values() for x in y]

# Parse words from file
words = []
with open('wordlist.txt', 'r') as f:
    words = [w.strip() for w in f.readlines()]

# Cache

if os.path.exists('data'):
    # Read cache
    with open('data', 'rb') as f:
        tfidf_w, tfidf_p, idx_w, model_p, syn = pickle.load(f)
else:
    # Calculate idf
    idf = {}
    cnt = {word: 0 for word in words}
    for poem in poems['all']:
        s = set()
        for l in poem['content']:
            for i in range(1, 4):
                for j in range(len(l)-i):
                    s.add(l[j:j+i])
        for word in s:
            if word in cnt:
                cnt[word] += 1

    for word in cnt:
        if cnt[word]>10:
            idf[word] = math.log10(len(poems['all'])/(1+cnt[word]))

    # Calculate tfidf (refer to About page for more infomation)
    tfidf_w = np.zeros((len(idf), len(poems['all'])), dtype=np.float64)
    tfidf_p = np.zeros((len(poems['all']), len(idf)), dtype=np.float64)
    idx_w = {}
    for idx, word in enumerate(idf):
        idx_w[word] = idx
    for idx_p, poem in enumerate(poems['all']):
        for l in poem['content']:
            cnt = 0
            for i in range(1, 4):
                for j in range(len(l)-i):
                    word = l[j:j+i]
                    if word in idf:
                        cnt += 1
            for i in range(1, 4):
                for j in range(len(l)-i):
                    word = l[j:j+i]
                    if word in idf:
                        tfidf_w[idx_w[word]][idx_p] += 1/cnt*idf[word]
                        tfidf_p[idx_p][idx_w[word]] += 1/cnt*idf[word]
    # Normalize vectors
    for v in tfidf_w:
        v /= np.linalg.norm(v)
    for v in tfidf_p:
        v /= np.linalg.norm(v)

    # PCA
    model_w = PCA(n_components=500)
    model_w.fit(tfidf_w)
    tfidf_w = model_w.transform(tfidf_w)

    model_p = PCA(n_components=500)
    model_p.fit(tfidf_p)
    tfidf_p = model_p.transform(tfidf_p)

    # Calculate synonyms
    def getSynonyms(key):
        a = []
        for word in idx_w:
            a.append((np.inner(tfidf_w[idx_w[key]], tfidf_w[idx_w[word]]), word))
        a.sort(reverse=True)
        r = a[0][0]
        return [(i/r, j) for i, j in a[:20]]

    syn = {}
    for w in idx_w:
        syn[w] = getSynonyms(w)

    with open('data', 'wb') as f:
        pickle.dump((tfidf_w, tfidf_p, idx_w, model_p, syn), f)



# Filter

# Method 1: Calculate the vector of keywords based on similarity between an arbitrary word and all keywords. Then sort the poems by the angles betwwen their tf-idf vectors and the vector of keywords.
def _filterPoems_tfidf(keywords, poems):
    if keywords==[]:
        return poems
    vec = np.zeros((len(idx_w),), dtype=np.float64)
    for word in idx_w:
        for k in keywords:
            v1, v2 = tfidf_w[idx_w[k]], tfidf_w[idx_w[word]]
            cos = np.inner(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
            cos = min(1, max(cos, -1))
            vec[idx_w[word]] += np.power(np.pi-np.arccos(cos), 5)
    vec /= np.linalg.norm(vec)
    vec = model_p.transform([vec])[0]
    a = []
    for idx_p, p in enumerate(poems):
        a.append((np.inner(tfidf_p[idx_p], vec), p))
    a.sort(key=lambda x: -x[0])
    return [x[1] for x in a[:20]], keywords

# Method 2: Extend the keywords to a larger set of similar words (refer to About page for more infomation). Then match a poem to the words to get a priority.
def _filterPoems_2(keywords, poems):
    wordWeight = {}
    for k in keywords:
        for w, word in syn[k]:
            wordWeight[word] = wordWeight.get(word, 0)+w
    a = []
    for p in poems:
        w = 0
        for l in p['content']:
            for i in range(1, 4):
                for j in range(len(l)-i):
                    if l[j:j+i] in wordWeight:
                        w += wordWeight[l[j:j+i]]
        a.append((w, p))
    a.sort(key=lambda x: -x[0])
    return [x[1] for x in a[:20]], list(wordWeight)

# Method 3: Straightforward maximal matching
def _filterPoems_3(keywords, poems):
    wordWeight = {word: 1 for word in keywords}
    a = []
    for p in poems:
        w = 0
        for l in p['content']:
            for i in range(1, 4):
                for j in range(len(l)-i):
                    if l[j:j+i] in wordWeight:
                        w += wordWeight[l[j:j+i]]
        a.append((w, p))
    a.sort(key=lambda x: -x[0])
    return [x[1] for x in a[:20]], list(wordWeight)

# General pattern of a poem filter.
def _filterPoems(keywords, poems):
    if keywords==[]:
        return poems, []
    return _filterPoems_2(keywords, poems)

# Handle author infomation of a query.
def filterPoems(x):
    x = x.strip().split(' ')
    if x[0] in poems:
        return _filterPoems(x[1:], poems[x[0]])
    else:
        return _filterPoems(x, poems['all'])

# Flask

subList = [['李白'], ['杜甫'], ['白居易'], ['王维'], ['李商隐'], ['孟浩然'], ['王昌龄'], ['温庭筠'], ['崔颢']]

app = Flask(__name__)
bootstrap = Bootstrap(app)
app.config['BOOTSTRAP_SERVE_LOCAL'] = True

@app.route("/")
def root():
    return render_template('index.html')

@app.route("/categories")
def categories():
    return render_template('categories.html')

@app.route("/manage_subscription")
def manage_subscription():
    return render_template('manage_subscription.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/_submit_subscription", methods=["POST"])
def _submit_subscription():
    subscription = request.form.get('sub').strip().split()
    if subscription in subList:
        return {'message': 'Fail: Duplicate subscription!', 'success': False}
    else:
        subList.insert(0, subscription)
        return {'message': 'Success!', 'success': True}

@app.route("/_unsubscribe", methods=["GET"])
def _unsubscribe():
    for x in subList:
        if ' '.join(x) == request.args.get('sub'):
            subList.remove(x)
            break
    return {}

@app.route("/_get_subList", methods=["GET"])
def _get_subList():
    return {'subList': [' '.join(x) for x in subList]}

@app.route("/_get_poems_by_sub", methods=["GET"])
def _get_poems_by_sub():
    poems, keywords = filterPoems(request.args.get('sub'))
    return {'poems': poems, 'keywords': keywords}


webbrowser.open('http://localhost:8080')
app.run(port=8080)