import sys
import time
import io
from pprint import pprint

import yaml

from ynlu import IntentClassifierClient

def getIUpairs(filename):
    intent, pos, neg = readYML(filename)
    pairs = pair(intent, pos)
    # test = pair(intent, neg)
    # print(intent)
    # print(pos)
    # print(neg)
    # print(pairs)
    # import ipdb; ipdb.set_trace()
    client = IntentClassifierClient(
        token='YOUR_TOKEN',
    )
    client.create_classifier('your_classifier_name')
    client.add_intent_utterance_pairs(pairs)
    client.train()
    print("training....")
    while True:
        if not client.classifier_is_traning():
            break        
        time.sleep(3)
    print("well...finished?")
    # import ipdb; ipdb.set_trace()
    result = client.predict('你會什麼')
    pprint(result)

def readYML(filename):
    intent = []
    pos = {}
    neg = {}
    with open(filename, 'r') as stream:
        inputVal = yaml.load(stream)
        # print(inputVal)
        for key, value in inputVal.items():
            intent.append(str(key))
            pos[str(key)] = value.get('positive')
            neg[str(key)] = value.get('negative')
            # print(str(elm))
    return intent, pos, neg

def pair(keyset, value):
    pairObj = []
    # import ipdb; ipdb.set_trace()
    firstKeyStr = 'intent'
    secondKeyStr = 'utterance'
    for key, val in value.items():
        for text in val:
            myDict = {}
            myDict[firstKeyStr] = key
            myDict[secondKeyStr] = text
            pairObj.append(myDict)
    return pairObj

def main():
    # input: <filename>
    if len(sys.argv) != 2:
        print("you dead wrong.")
        return
    getIUpairs(sys.argv[1])

if __name__ == '__main__':
    main()
