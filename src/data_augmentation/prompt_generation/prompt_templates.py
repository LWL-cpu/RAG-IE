

def get_zero_shot_template(instance):
    """ Get zero shot template
    Args:
        sentence: input sentence
        relation: relation type
    return: zero shot template
    """

    template_zero_shot = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + \
                         "Examples:" + '\n' + \
                         'Document: Transportation officials are urging carpool and teleworking as options to combat an expected flood of drivers on the road . ( Paul Duggan ) -- A Baltimore prosecutor accused a police detective of “ sabotaging ” investigations related to the death of Freddie Gray , accusing him of fabricating notes to suggest that the state ’s medical examiner believed the manner of death was an accident rather than a homicide . The heated exchange came in the chaotic sixth day of the trial of Baltimore Officer Caesar Goodson Jr. , who drove the police van in which Gray suffered a fatal spine injury in 2015 . ( Derek Hawkins and Lynh Bui )\nEvent: life.die.deathcausedbyviolentevents\nTrigger: homicide\nPossible roles: instrument, killer, victim, place\nArguments: [{"type": "killer", "argument": "Officer Caesar Goodson Jr."}, {"type": "victim", "argument": "Freddie Gray"}, {"type": "place", "argument": "Baltimore"}]\nExamples end here.\n' + \
                         "Question: " + '\n' + \
                         instance + \
                         'Arguments:'
    return template_zero_shot

def get_zero_shot_template_rag(instance, context, topk=1):
    """ Get rag template
    Args:
        sentence: input sentence
        relation: relation type
    return: rag template
    """
    context = context[:topk]
    context = "\n".join(context)
    template_zero_shot = """Task description: Given a document and an event, you need to identify all arguments of this event, and classify the role of this argument. Limit responses to arguments only.  Please answer in JSON format of [{"type": <role>, "argument": <argument>}, {"type": <role>, "argument": <argument>}, ...].\n""" + \
                         "Examples:" + '\n' + context + '\n' + \
                         'Document: Transportation officials are urging carpool and teleworking as options to combat an expected flood of drivers on the road . ( Paul Duggan ) -- A Baltimore prosecutor accused a police detective of “ sabotaging ” investigations related to the death of Freddie Gray , accusing him of fabricating notes to suggest that the state ’s medical examiner believed the manner of death was an accident rather than a homicide . The heated exchange came in the chaotic sixth day of the trial of Baltimore Officer Caesar Goodson Jr. , who drove the police van in which Gray suffered a fatal spine injury in 2015 . ( Derek Hawkins and Lynh Bui )\nEvent: life.die.deathcausedbyviolentevents\nTrigger: homicide\nPossible roles: instrument, killer, victim, place\nArguments: [{"type": "killer", "argument": "Officer Caesar Goodson Jr."}, {"type": "victim", "argument": "Freddie Gray"}, {"type": "place", "argument": "Baltimore"}]\nExamples end here.\n' + \
                         "Question:" + '\n' + \
                         instance + \
                         'Arguments:'
    return template_zero_shot

def get_zero_shot_template_memory(instance, dataset, topk=1):
    """ Get rag template
    Args:
        sentence: input sentence
        relation: relation type
    return: rag template
    """
    template_zero_shot = """You are a specialist in event extraction. Given a document and an event, please identify all arguments of this event according to given possible roles, and classify the role of each argument.  Limit responses to arguments only.  Please answer in the format of (type: <role>, argument: <argument>).\n""" + \
                         "Examples:" + '\n' + \
                         'Question:\nDocument: Transportation officials are urging carpool and teleworking as options to combat an expected flood of drivers on the road . ( Paul Duggan ) -- A Baltimore prosecutor accused a police detective of “ sabotaging ” investigations related to the death of Freddie Gray , accusing him of fabricating notes to suggest that the state ’s medical examiner believed the manner of death was an accident rather than a homicide . The heated exchange came in the chaotic sixth day of the trial of Baltimore Officer Caesar Goodson Jr. , who drove the police van in which Gray suffered a fatal spine injury in 2015 . ( Derek Hawkins and Lynh Bui )\nEvent: life.die.deathcausedbyviolentevents\nTrigger: homicide\nPossible roles: instrument, killer, victim, place\nArguments: ( type: killer, argument: Officer Caesar Goodson Jr. ), ( type: victim, argument: Freddie Gray ), ( type: place, argument: Baltimore )\n' + \
                         "Question: " + '\n' + \
                         instance + \
                         'Arguments:'
    return template_zero_shot, dataset


# def get_zero_shot_template_tacred(sentence, relation, head, tail):
#     """ Get zero shot template
#     Args:
#         sentence: input sentence
#         relation: relation type
#     return: zero shot template
#     """
#     template_zero_shot ="""Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n""" +\
#                         """Question : What is the relation type between head and tail entities in the following sentence?\n""" +\
#                         """ Sentence:""" + str(sentence)+ """\n""" +\
#                         """ Head entity: """ + head + """. \n""" +\
#                         """ Tail entity: """ + tail + """. \n""" +\
#                         """ Relation types: """ + str(relation) + """. \n""" +\
#                         """ output format: relation_type\n""" +  "Please directly answer the relation_type between the head and tail entities from the following relation list \n" +  \
#                         """ Relation types: """ + str(relation) + """. \n""" +\
#                         '''Answer: '''
#     return template_zero_shot

# def get_zero_shot_template_tacred_rag(sentence, relation, head, tail, context, topk):
#     """ Get rag template
#     Args:
#         sentence: input sentence
#         relation: relation type
#     return: rag template
#     """
#     context = "\n".join(context[:topk])
#     template_zero_shot = """Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n""" +\
#                         """ Question : What is the relation type between tail and head entities according to given relationships below in the following sentence?\n""" +\
#                         """ Query Sentence:""" + str(sentence)+ """\n""" +\
#                         """ Head entity: """ + head + """. \n""" +\
#                         """ Tail entity: """ + tail + """. \n""" +\
#                         """ output format: relation_type\n""" + \
#                         """ Examples: """+ str(context)+ """\n""" +\
#                           "Please directly answer the relation_type between the head and tail entities from the following relation list \n" +  \
#                         """ Relation types: """ + str(relation) + """. \n""" +\
#                         '''Answer: '''
#     return template_zero_shot

def semeval_prompt_template_rag(sentence, relation, head, tail, head_name, tail_name, context, topk):
    """ Get rag template
    Args:
        sentence: input sentence
        relation: relation type
    return: rag template
    """
    context = "\n".join(context[:topk])
    template_zero_shot = """Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n Please directly answer the relation_type of the following question.\n""" +\
                        """ Examples:"""+str(context)+ """\n""" + \
                        """ Question : What is the relation type between """+head+""" and """+tail+""" entities according to given relationships below in the following sentence, considering example sentence and its relationship?\n""" +\
                        """ Query Sentence:""" + str(sentence)+ """\n""" +\
                        """ e1: """ + head_name + """. \n""" +\
                        """ e2 : """ + tail_name + """. \n""" +\
                        """ Relation types: """ + str(relation) + """. \n""" +\
                        """ Output format: relation_type.\n"""  + \
                        " Please directly answer the relation_type\n"  + \
                        '''Answer:'''
    return template_zero_shot
def semeval_prompt_template(sentence, relation, head, tail, head_name, tail_name):
    """ Get zero shot template
    Args:
        sentence: input sentence
        relation: relation type
    return: zero shot template
    """
    template_zero_shot = """Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n Please directly answer the relation_type of the following question.\n""" +\
    """Question : What is the relation type between """+head+""" and """+tail+""" entities in the following sentence?\n""" +\
                        """ Sentence:""" + str(sentence)+ """\n""" +\
                        """ e1: """ + head_name + """. \n""" +\
                        """ e2 : """ + tail_name + """. \n""" +\
                        """ Relation types: """ + str(relation) + """. \n""" +\
                        """ output format: relation_type""" +  " Please directly answer the relation_type\n" +  \
                        '''Answer:'''
    return template_zero_shot


def get_zero_shot_template_tacred(sentence, relation, head, tail):
    """ Get zero shot template
    Args:
        sentence: input sentence
        relation: relation type
    return: zero shot template
    """
    template_zero_shot ="""Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n""" +\
                        """Question : What is the relation type between head and tail entities in the following sentence?\n""" +\
                        """ Sentence:""" + str(sentence)+ """\n""" +\
                        """ Head entity: """ + head + """. \n""" +\
                        """ Tail entity: """ + tail + """. \n""" +\
                        """ Relation types: """ + str(relation) + """. \n""" +\
                        """ Output format: relation_type\n""" +  "Please directly answer the relation_type between the head and tail entities from the following relation list \n" +  \
                        """ Relation types: """ + str(relation) + """. \n""" +\
                        '''Answer: '''
    return template_zero_shot

def get_zero_shot_template_tacred_rag(sentence, relation, head, tail, context, topk):
    """ Get rag template
    Args:
        sentence: input sentence
        relation: relation type
    return: rag template
    """
    context = "\n".join(context[:topk])
    template_zero_shot = """Problem Definition: Relation extraction is to identify the relationship between two entities in a sentence.\n""" +\
                        """ Examples: """+ str(context)+ """\n""" +\
                        """ Question : What is the relation type between tail and head entities according to given relationships below in the following sentence?\n""" +\
                        """ Query Sentence:""" + str(sentence)+ """\n""" +\
                        """ Head entity: """ + head + """. \n""" +\
                        """ Tail entity: """ + tail + """. \n""" +\
                        """ Output format: relation_type\n""" + \
                          "Please directly answer the relation_type between the head and tail entities from the following relation list \n" +  \
                        """ Relation types: """ + str(relation) + """. \n""" +\
                        '''Answer:'''
    return template_zero_shot