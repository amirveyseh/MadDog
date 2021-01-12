import json
import re
import spacy

nlp = spacy.load("en_core_web_sm")

from prototype.acronym.utils import constant
from prototype.acronym.zeroshot.zeroshot import ZeroShotExtractor
from prototype.acronym.rulebasedExtractor import Extractor

zeroshot_extractor = ZeroShotExtractor()
ruleExtractor = Extractor()

diction_unamb = {}

class AcronymPredictor():
    def predict(self, sentence, use_zeroshot=False):
        # shorts = shortPredictor.predict(sentence)
        pred_PATH = '../acronym/predict.json'
        tokens = [t.text for t in nlp(sentence) if len(t.text.strip()) > 0]
        # FIXED [issue: We show that stochastic gradient Markov chain Monte Carlo ( SG - MCMC ) - a class of ]
        if constant.RULES['merge_hyphened_acronyms']:
            tokens, _ = ruleExtractor.merge_hyphened_acronyms(tokens)
        rulebased_pairs = ruleExtractor.extract(tokens, constant.RULES)
        zeroshot_pairs = []
        if use_zeroshot:
            zeroshot_pairs = zeroshot_extractor.extract(tokens)
        datas = []
        for i, t in enumerate(tokens):
            detector = []
            if t in rulebased_pairs and len(t) > 1 and not t.isdigit() and bool(re.search('[a-zA-Z]', t)):
                detector += ['pipeline extractor']
            # if t in diction and len(t) > 1 and not t.isdigit() and bool(re.search('[a-zA-Z]', t)):
            #     detector += ['string-match']
            if t in diction_unamb and len(t) > 1 and not t.isdigit() and bool(re.search('[a-zA-Z]', t)) and len([c for c in t if bool(re.search('[a-zA-Z]', c))]) > 1:
                detector += ['dictionary lookup']
            if i in zeroshot_pairs and len(t) > 1 and not t.isdigit() and bool(re.search('[a-zA-Z]', t)):
                detector += ['zeroshot']
            if len(detector):
                datas.append({
                    'token': tokens,
                    'acronym': i,
                    'expansion': 'National_Broadcasting_Company',
                    'detector': ' & '.join(detector)
                })
        if len(datas) == 0:
            return [], tokens
        elif len(datas) == 1:
            datas.append(datas[0])
        with open(pred_PATH, 'w') as file:
            json.dump(datas, file)
        predictions = dict([(datas[i]['acronym'],[datas[i]['detector'], {'disambiguator': 'NOT-SUPPORTED'}]) for i in range(len(datas))])
        for acr, pred in predictions.items():
            if acr in zeroshot_pairs:
                pred[1]['zeroshot'] = zeroshot_pairs[acr]
        for acr, pred in predictions.items():
            if tokens[acr] in rulebased_pairs:
                pred[1]['rule-based'] = (rulebased_pairs[tokens[acr]][0], rulebased_pairs[tokens[acr]][3])
        for acr, pred in predictions.items():
            if tokens[acr] in diction_unamb:
                pred[1]['dictionary'] = diction_unamb[tokens[acr]]
        return predictions,tokens