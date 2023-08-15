from transformers import AutoTokenizer, LongT5ForConditionalGeneration
import re
import json
import evaluate
#from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu
from argparse import ArgumentParser


def parse_DOT(dot_str):
    # output: graph edge list
    if len(dot_str) < 15:
        return [], 0

    edges = dot_str[15:].split(';')
    key_set = set()
    graph = [] # graph edge list
    duplicate = 0
    for edge_str in edges:
        edge_str = re.sub('is_included', 'isincluded', edge_str)
        rel_list = re.findall(r'rel=([a-zA-Z]+)', edge_str)
        if len(rel_list) < 1:
            break
        rel = rel_list[0].lower()
        if rel not in ['after', 'before', 'includes', 'simultaneous', 'isincluded']: #['after', 'before']:
            continue
        if rel == 'isincluded':
            rel = 'is_included'
        event_pair = edge_str.split('[rel=')[0]
        if len(event_pair.split('--')) < 2:
            continue
        event_1 = event_pair.split(' -- ')[0].lower()
        event_2 = event_pair.split(' -- ')[1].lower()
        if event_1[0] == ' ':
            event_1 = event_1[1:]
        event_1 = re.sub(r'\"', '', event_1)
        event_2 = re.sub(r'\"', '', event_2)
        event_1 = re.sub(r'\n', '', event_1)
        event_2 = re.sub(r'\n', '', event_2)
        if len(event_1)==0 or len(event_2)==0:
            continue
        if event_1==" " or event_2==" ":
            continue
        if event_1[0] == ' ':
            event_1 = event_1[1:]
        if event_2[0] == ' ':
            event_2 = event_2[1:]
        if event_1[-1] == ' ':
            event_1 = event_1[:-1]
        if event_2[-1] == ' ':
            event_2 = event_2[:-1]

        key = f"{event_1}||{rel}||{event_2}"
        if key in key_set:
            duplicate += 1
        else:
            graph.append((event_1, rel, event_2))
            key_set.add(key)
            #print(event_1, rel, event_2)
    #print(f"Num of duplication: {duplicate}")
    return graph, duplicate


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str, default="long-t5-tglobal-base-finetuned-NYT_aug5-final")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = LongT5ForConditionalGeneration.from_pretrained(args.model_checkpoint, device_map="auto")

    rouge = evaluate.load('rouge')
    graph_string_metrics = {}
    
    with open("data/NYT_des_test.json", 'r') as f:
        nyt_dev = json.loads(f.read())

    total_generated = 0
    total_gold = 0
    test_out = {}
    predictions = []
    references = []
    total_duplicate = 0
    for doc_id in nyt_dev:
        test_doc = nyt_dev[doc_id]['document']
        gold_graph, _ = parse_DOT(nyt_dev[doc_id]['target'])
        total_gold += len(gold_graph)

        outputs = model.generate(tokenizer.encode(test_doc, return_tensors="pt").cuda(), max_length=512)
        out_str = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # graph string metircs
        predictions.append(out_str.lower())
        references.append(nyt_dev[doc_id]['target'].lower())

        generated_graph, duplicate = parse_DOT(out_str)
        total_duplicate += duplicate
        total_generated += len(generated_graph)
        test_out[doc_id] = {
            "document": nyt_dev[doc_id]['document'],
            "gold": gold_graph,
            "generated": generated_graph,
            "out_string": out_str
        }

    graph_string_metrics["NYT_dev"] = rouge.compute(predictions=predictions, references=references)
    graph_string_metrics["NYT_dev"]['BLEU'] = corpus_bleu([[graph_str.split()] for graph_str in references], [graph_str.split() for graph_str in predictions])

    print(f"Total generated: {total_generated}\nTotal gold: {total_gold}\nTotal duplicate: {total_duplicate}")
    with open(f'{args.model_checkpoint}-nyt-dev.json', 'w') as f:
        f.write(json.dumps(test_out, indent=4))
    
    print(graph_string_metrics)
    with open('graph_string_metrics.json', 'w') as f:
        f.write(json.dumps(graph_string_metrics, indent=4))



