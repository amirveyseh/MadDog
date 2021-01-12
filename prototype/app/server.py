from flask import Flask, request, render_template, redirect
import time
import urllib.parse
from collections import OrderedDict

from prototype.acronym.predict import AcronymPredictor

acronym_predictor = AcronymPredictor()
# definition_predictor = DefinitionPredictor()

app = Flask(__name__)


def deep_strip(text):
    new_text = ''
    for c in text:
        if len(c.strip()) > 0:
            new_text += c
        else:
            new_text += ' '
    new_text = new_text.replace('"', '\'')
    return new_text


@app.route('/')
def my_form():
    return render_template('form.html')


@app.route('/annotate')
def annotate_func():
    sentence = urllib.parse.unquote(request.args.get('sent'))
    res = render_template('annotate.html', content=sentence)
    return res

@app.route('/feedback', methods=['POST'])
def save_feedback_func():
    text = deep_strip(request.form['text'])
    sent = request.form['sent'].replace('>>', ' ')
    if 'incorrect' in request.form:
        incorrect = request.form['incorrect']
    else:
        incorrect = ''
    if 'public' not in request.form:
        public = False
    else:
        public = True
    with open('feedback.txt', 'a') as file:
        file.write(text + '\n')
        file.write('-' * 25 + '\n')
        file.write(sent + '\n')
        file.write('-' * 25 + '\n')
        file.write(str(incorrect == 'TRUE') + '\n')
        file.write('-' * 25 + '\n')
        file.write(str(public) + '\n')
        file.write('=' * 50 + '\n')
    return redirect("/")


@app.route('/save_annotation')
def save_annotate_func():
    annotations = urllib.parse.unquote(request.args.get('annotations'))
    mapping = urllib.parse.unquote(request.args.get('mapping'))
    sent = urllib.parse.unquote(request.args.get('sent')).replace('&gt;&gt;', ' ')
    with open('annotations.txt', 'a') as file:
        file.write(annotations + '\n')
        file.write('-' * 25 + '\n')
        file.write(mapping + '\n')
        file.write('-' * 25 + '\n')
        file.write(sent + '\n')
        file.write('=' * 50 + '\n')
    return redirect("/")


@app.route('/', methods=['POST'])
def my_form_post():
    text = deep_strip(request.form['text'])
    #with open('input.txt') as file:
    #    text = file.read()
    print(text)
    before_time = int(round(time.time() * 1000))
    acronym_predictions, tokens = acronym_predictor.predict(text,'zeroshot' in request.form)
    after_time = int(round(time.time() * 1000))
    elapsed_time = after_time - before_time
    # definition_predictions = definition_predictor.predict(text)
    glossary = {}
    res = ''
    for i, t in enumerate(tokens):
        if i in acronym_predictions:
            alert_text = ''
            system_pred = ''
            if 'rule-based' in acronym_predictions[i][1] and acronym_predictions[i][1]['rule-based'][0]:
                system_pred = acronym_predictions[i][1]['rule-based'][0].replace("'",'')
                glossary[t] = (system_pred, 'Pipeline ('+acronym_predictions[i][1]['rule-based'][1]+')')
            if not system_pred and 'dictionary' in acronym_predictions[i][1] and acronym_predictions[i][1]['dictionary']:
                system_pred = acronym_predictions[i][1]['dictionary'].replace("'", '')
                glossary[t] = (system_pred, 'Dictionary Lookup')
            if not system_pred and acronym_predictions[i][1]['disambiguator'] != 'NOT-SUPPORTED':
                system_pred = acronym_predictions[i][1]['disambiguator'].replace("'", '')
                glossary[t] = (system_pred, 'Supervised Disambiguator')
            if not system_pred and 'extractor' in acronym_predictions[i][1] and acronym_predictions[i][1]['extractor']:
                system_pred = acronym_predictions[i][1]['extractor'].replace("'", '')
                glossary[t] = (system_pred, 'Supervised Extractor')
            if not system_pred and 'zeroshot' in acronym_predictions[i][1] and acronym_predictions[i][1]['zeroshot']:
                system_pred = acronym_predictions[i][1]['zeroshot'][0].replace("'", '')
                glossary[t] = (system_pred, 'Zero-shot')

            alert_text += 'System Prediction: ' + system_pred + '\\n\\n' + '-'*20 + '\\n\\nMore info:\\n'

            if acronym_predictions[i][1]['disambiguator'] != 'NOT-SUPPORTED':
                alert_text += 'Supervised Disambiguator: ' + acronym_predictions[i][1]['disambiguator'].replace("'", '')
            if 'rule-based' in acronym_predictions[i][1] and acronym_predictions[i][1]['rule-based'][0]:
                alert_text += '\\nPipeline: ' + acronym_predictions[i][1]['rule-based'][0].replace("'",'') + " (method: " + \
                              acronym_predictions[i][1]['rule-based'][1] + ")"
            if 'extractor' in acronym_predictions[i][1] and acronym_predictions[i][1]['extractor']:
                alert_text += '\\nSupervised Extractor: ' + acronym_predictions[i][1]['extractor'].replace("'", '')
            if 'dictionary' in acronym_predictions[i][1] and acronym_predictions[i][1]['dictionary']:
                alert_text += '\\nDictionary Lookup: ' + acronym_predictions[i][1]['dictionary'].replace("'", '')
            if 'zeroshot' in acronym_predictions[i][1] and acronym_predictions[i][1]['zeroshot']:
                alert_text += '\\nDisambiguation: ' + acronym_predictions[i][1]['zeroshot'][0].replace("'", '')
                alert_text += '\\n\\nMeanings:'
                for meaning in acronym_predictions[i][1]['zeroshot'][1]:
                    alert_text += '\\n'+meaning[0].replace("'", '')+" : "+str(meaning[1])
            detected_by = acronym_predictions[i][0]
            if 'zero' in detected_by.lower():
                detected_by = 'Disambiguation Model'
            alert_text += '\\n\\nDetected by: ' + detected_by
            res += '<span style="background-color: pink; cursor: pointer; font-weight: bold;", onclick="alert(\'' + alert_text + '\')"> ' + t + ' </span>'
        # elif i in definition_predictions:
        #     res += '<span style="background-color: pink; cursor: pointer; font-weight: bold;", onclick="alert(\'' + definition_predictions[
        #         i] + '\')"> ' + t + ' </span>'
        else:
            res += '<span> ' + t + ' </span>'
    res += '<br/><br/><a href="/"><button>Return</button></a>'
    res += '<br><br>Processing Time: '+str(elapsed_time)+' ms <br>'
    res += '<br/>Glossary: <br><br>'
    glossary = OrderedDict(sorted(glossary.items(), key=lambda t: t[0]))
    for acr, info in glossary.items():
        detector = info[1]
        if 'zero' in detector.lower():
            detector = 'Disambiguation Model'
        res += acr + ' : ' + info[0] + ' (detected by: ' + detector+')' + '<br>'
    res += '<br/>'
    res += '<form method="POST" action="/feedback">'
    res += '<br><input input type="radio" id="incorrect" name="incorrect" value="TRUE"><label for="incorrect"> Incorrect prediction? </label><br><br>'
    res += '<br><input input type="radio" id="correct" name="incorrect" value="FALSE"><label for="correct"> Correct prediction? </label><br><br>'
    res += '<br><input input type="checkbox" id="public" name="public" checked><label for="public"> This text can be publicly released </label><br><br>'
    res += '''
        <label for="text">Feedback:</label>
        <textarea id="text" name="text" rows="4" cols="50"></textarea>
    '''
    res += '<input type="hidden" name="sent" value="' + '>>'.join(tokens) + '">'
    res += '<br><br><input type=submit>'
    res += '</form>'
    res += '<br><br>Do you want to try it yourself? Click <a href="/annotate?sent=' + urllib.parse.quote(
        '>>'.join(tokens)) + '"> here <a>' + ' to annotate this sample.'

    #with open('output.html', 'w') as file:
    #    file.write(res)
    return res

if __name__ == '__main__':
    app.run(host='0.0.0.0')
