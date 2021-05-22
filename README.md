# tsvar

Association rules for TSV (tab-separated values) files.

[Association rules](https://en.wikipedia.org/wiki/Association_rule_learning) describe the relationship between two categorical variables. They are represented in the form `A -> B`, where being/having A (the antecedent) increases the probability of being/having B (the consequent). A and B can be sets. For more details, see [mlxtend's overview](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/).

## Dependencies

Python 3.7+ (probably works on lower versions), Pandas and Mlxtend.

## Primer

TsvAR is a generalization of the association miner in [bandori-2019-stats](https://github.com/supreme-chocomint/bandori-2019-stats). Therefore (like it's predecessor), it can create rules from input survey response data, such as:

```
| Favorite Operators            | Gender | Region        |
| ----------------------------- | ------ | ------------- |
| Beanstalk,Blue Poison,Phantom | Male   | Europe        |
| Scene,Astesia,Texas,Hellagur  | Other  | East Asia     |
| Perfumer,Mountain,Blemishine  | Female | Latin America |
```

However, it can handle any data where each person (or thing) has a closed-set of categorical properties (e.g. social media profiles, networks in general), such as:

```
| Friends                  | Hobbies                      | Age Group | Invited |
| ------------------------ | ---------------------------- | --------- | ------- |
| Alice,Bob,Claris         | Gardening,Hiking,Board Games | 18-22     | No      |
| Derek,Ethan,Fumino,Alice | Reading                      | 26-30     | No      |
| Giovanni,Claris          | Movies,Soccer,Hiking         | 18-22     | Yes     |
```

## Usage

In `path-to-file.tsv`:
```
Favorite Operators	Gender	Region
Beanstalk,Blue Poison,Phantom	Male	Europe
Scene,Astesia,Texas,Hellagur	Other	East Asia
Perfumer,Mountain,Blemishine	Female	Latin America
```

In your code:
```
from tsvar import AssociationMiner

miner = AssociationMiner("path-to-file.tsv", export_to_tsv=True)
rules = miner.mine(["Favorite Operators", "Region"], min_frequency=0.01, metric="confidence", metric_threshold=0.3)
```

This will return association rules regarding both Favorite Operators and/or Region values that occur in at least 1% of all entries and have a confidence of at least 30%. It will also export those rules as a TSV file, which you can open in Excel etc.

To search for rules based on specific value occurrences, or reorganize the rules, do something like this:

```
resultant_rules_df = rules.search(["Blemishine"], location="antecedent")

rules.organize(sort_by="support")
organized_rules_df = rules.table_organized
```

To get a list of unique values for a category:

```
import pandas
from tsvar import Helper

df = pandas.read_table("path-to-file.tsv")
answers = Helper.unique_answers(df, "Hobbies")
```

In general, see inline documentation for more options, examples, and explanations.

