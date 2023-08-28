import re


def make_prediction(rules, classification_model, rule_model, data):
    result = 0
    rules_covering_data = 0
    for i in range(len(rules)):
        if bool(re.match(rules[i], data)):
            rules_covering_data += 1
            result += rule_model(data)[i]
    result /= rules_covering_data
    result += classification_model(data)[1]
