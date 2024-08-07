REL_TO_PHRASE = {
    "oEffect":"effects on PersonY are",
    "oReact":"then, PersonY feels",
    "oWant":"then, PersonY wants",
    "xAttr":"So, PersonX is seen as",
    "xEffect":"effects on PersonX are",
    "xIntent":"because PersonX intended",
    "xNeed":"before that PersonX requires",
    "xReact":"then, PersonX feels",
    "xWant":"then, PersonX wants",
    'isAfter':"is after", 
    'isBefore':"is before", 
    "mnli":"The relation between sentences is",
    "rte":"The relation between sentences is",
    "qnli":"The relation between sentences is",
    "stsb":"The similarity between sentences is",
    "sst2":"The sentiment of sentence is",
    "mrpc":"The sentences are",
    "qqp":"The sentences are",
    "grammatical":"The validity of stence is",
    'AtLocation':"is located at", 
    'ObjectUse':"is used for", 
    'Desires': " desires ",
    'HasProperty':"has ", 
    'NotDesires': "doesn't like", 
    'Causes':"causes", 
    'HasSubEvent':"occurs when", 
    'xReason':"is a reason for",
    'CapableOf':"capable of", 
    'MadeUpOf':"is made of", 
    'isAfter':"comes after", 
    'isBefore':"comes before", 
    'isFilledBy': "is filled by",
    'HinderedBy':"is hindered by",
}
REL_TO_SHARED_TOKENS = {
    "mnli":"mnli qnli rte",
    "rte":"mnli qnli rte",
    "qnli":"mnli qnli rte",
    "sst2":"mrpc sst2 qqp",
    "mrpc":"mrpc sst2 qqp",
    "qqp":"mrpc sst2 qqp",
    "superglue-wic":"cls wic syn ss",
    "oEffect":"event after others effect",
    "oReact":"adj after others react",
    "oWant":"event after others want",
    "atomic-rels":"event after they want",
    "xAttr":"adj always person seen",
    "xEffect":"event after person effect",
    "xIntent":"event before person want",
    "xNeed":"event before person need",
    "xReact":"adj after they react",
    "xWant":"event after they want",
    'AtLocation':"is located", 
    'ObjectUse':"is used for", 
    'Desires': "person desire event always",
    'HasProperty':"has property", 
    'NotDesires': "no_desire", 
    'Causes':"event causes event reason", 
    'HasSubEvent':"<has_sub>", 
    'xReason':"event reason for event",
    'CapableOf':"<capable>", 
    'MadeUpOf':"<madeof>", 
    'isAfter':"<isAfter>", 
    'isBefore':"<isBefore>", 
    'isFilledBy': "<isFilledBy>",
    'HinderedBy':"<HinderedBy>",
    'cola':"Is it grammatical or meaningful",
}
REL_TO_WORD = {
    "oEffect":"effect on others",
    "oReact":"reaction of others",
    "oWant":"others want",
    "xAttr":"seen",
    "xEffect":"effect on them",
    "xIntent":"intend",
    "xNeed":"need",
    "xReact":"person reaction",
    "xWant":"want",
    'AtLocation':"location", 
    'ObjectUse':"use", 
    'Desires': "desire",
    'HasProperty':"property", 
    'NotDesires': "no_desire", 
    'Causes':"cause", 
    'HasSubEvent':"sub event", 
    'xReason':"reason",
    'CapableOf':"capable", 
    'MadeUpOf':"made of", 
    'isAfter':"is after", 
    'isBefore':"is before", 
    'isFilledBy': "is filled by",
    'HinderedBy':"hindered by",
    'cola':"Is it grammatical or meaningful",
}
REL_TO_TOKEN = {
    "cb":"<cb>",
    "oEffect":"<oEffect>",
    "oReact":"<oReact>",
    "oWant":"<oWant>",
    "xAttr":"<xAttr>",
    "xEffect":"<xEffect>",
    "xIntent":"<xIntent>",
    "xNeed":"<xNeed>",
    "xReact":"<xReact>",
    "xWant":"<xWant>",
    'AtLocation':"<loc>", 
    'ObjectUse':"<use>", 
    'Desires': "<desire>",
    'HasProperty':"<prop>", 
    'NotDesires': "<no_desire>", 
    'Causes':"<cause>", 
    'HasSubEvent':"<has_sub>", 
    'xReason':"<xReason>",
    'CapableOf':"<capable>", 
    'MadeUpOf':"<madeof>", 
    'isAfter':"<isAfter>", 
    'isBefore':"<isBefore>", 
    'isFilledBy': "<isFilledBy>",
    'HinderedBy':"<HinderedBy>"
}
GEN_TOKENS = {
    "en": "<gen_en>",
    "fa": "<gen_fa>"
}
