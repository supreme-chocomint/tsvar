from functools import wraps

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder


def _can_export(f):
    """
    Decorator for AssociationMiner methods that return Rules.
    If AssociationMiner is set to export to TSV, then exporting occurs.
    :param f: Method
    :return: Method
    """

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        res = f(self, *args, **kwargs)  # Rules object
        if self.export_to_tsv:
            name = "mine_results"
            if res.table_organized is not None:
                res.table_organized.to_csv(f"{name}.organized.tsv", sep="\t")
            res.table.to_csv(f"{name}.tsv", sep="\t")
        return res

    return wrapper


class Helper:

    @staticmethod
    def unique_answers(df, column):
        """
        Finds all unique answers from all responses.
        Main use case is to break up multi-answer responses into individual answers,
        but this method works on single-answer responses as well (it just does unnecessary work).
        Substrings assumed to be individual answers if comma-separated (after removing round brackets and their
        contents).
        e.g. if response is "Europe (includes Russia), North America [NA] (includes Mexico, Central America,
                Caribbean)", then "Europe" and "North America [NA]" are the two individual answers of the response,
                and will be included in the returned array.
        :return: Array
        """
        # remove round brackets' contents (https://stackoverflow.com/a/40621332)
        df[column].replace(r"\([^()]*\)", "", regex=True, inplace=True)
        answers = df[column].str.split(",", expand=True)  # split up answers
        return answers.stack().str.strip().unique().tolist()  # make into Series, clean, and get all unique


class AssociationMiner:
    """
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/
    """

    def __init__(
            self,
            tsv_path,
            export_to_tsv=False
    ):
        self.df = pd.read_table(tsv_path)
        self.export_to_tsv = export_to_tsv

    @_can_export
    def mine(
            self,
            columns,
            min_frequency=0.001,
            metric="lift",
            metric_threshold=1.1
    ):
        """
        Generic function to mine rules from responses. Default metric is lift > 110%.
        If confidence is too high, rules are too specific to individual people, as opinions vary quite a bit.
        :param columns: List of column names to consider.
        :param min_frequency: threshold frequency for itemset to be considered "frequent"
        :param metric: "support", "confidence", "lift", "leverage", or "conviction"
        :param metric_threshold: Float, valid values depend on metric
        :return: Rules
        """
        raw_itemsets = self._generate_frequent_itemsets(columns, min_frequency)
        return self._generate_association_rules(raw_itemsets, metric, metric_threshold)

    def _generate_frequent_itemsets(
            self,
            columns,
            min_frequency
    ):
        """
        Uses the values of columns to generate frequent itemsets for association rule mining.
        :param columns: List of column names to use
        :param min_frequency: threshold frequency for set to be considered "frequent"
        :return DataFrame
        """
        lists = self._reduce(self.df, columns)
        one_hot_df = self._transform_to_one_hot(lists)
        return self._find_sets(one_hot_df, min_frequency=min_frequency)

    def _generate_association_rules(
            self,
            itemsets,
            metric,
            metric_threshold
    ):
        """
        Uses frequent itemsets to generate rules with 1 antecedent and sorted by lift.
        :param itemsets: DataFrame
        :param metric: "support", "confidence", "lift", "leverage", or "conviction"
        :param metric_threshold: Float
        :return: Rules
        """
        rules = self._find_rules(itemsets, metric, metric_threshold)
        rules.organize(max_antecedents=1, max_consequents=1, sort_by=["lift"], sort_ascending=[False])
        return rules

    @staticmethod
    def _find_sets(
            one_hot_df,
            min_frequency
    ):
        """
        Finds frequent itemsets.
        :param min_frequency: Float; threshold occurrence for a set to be considered "frequent"
        :return DataFrame
        """
        itemsets = fpgrowth(one_hot_df, min_support=min_frequency, use_colnames=True, max_len=2)
        print("Done generating itemsets.")
        return itemsets.sort_values(by=["support"], ascending=False)

    @staticmethod
    def _find_rules(
            itemsets,
            metric,
            metric_threshold
    ):
        """
        Uses itemsets attribute to find rules.
        :param metric: e.g. "confidence", "lift"
        :param metric_threshold: Float
        :return Rules
        """
        return Rules(association_rules(itemsets, metric=metric, min_threshold=metric_threshold))

    @staticmethod
    def _reduce(
            df,
            column_list
    ):
        """
        Reduces a DataFrame to lists, where each list holds the values of the columns listed in column_list.
        :param df: DataFrame
        :param column_list: A list of columns to reduce to
        :return List of Lists
        """
        rows = []

        # Make rows
        for _ in range(len(df)):
            rows.append([])

        # Populate rows
        for column in column_list:
            for row_i, column_value in df[column].items():
                # Get all actual values in comma-separated multi-response
                rows[row_i].extend(column_value.split(","))

        return rows

    def _transform_to_one_hot(
            self,
            itemset_list
    ):
        """
        Converts itemset list into one-hot encoded DataFrame,
        which is required for frequent itemset mining.
        :param itemset_list: A list of lists
        :return DataFrame
        """
        encoder = TransactionEncoder()
        array = encoder.fit(itemset_list).transform(itemset_list)
        df = pd.DataFrame(array)

        # rename columns
        columns = self._parse_columns(encoder.columns_)
        df.rename(columns={k: v for k, v in enumerate(columns)}, inplace=True)

        return df

    @staticmethod
    def _parse_columns(
            columns
    ):
        """
        Remove quotes in column names, because Pandas doesn't like them
        """
        res = []
        for column in columns:
            res.append(column.replace('"', ''))
        return res


class Rules:
    """
    Represents a set of association rules.
    """

    def __init__(
            self,
            df
    ):
        """
        :param df: DataFrame; original table that is always retained
        """
        self._df = df
        self._organized_df = None
        self._sort_by = ["lift"]
        self._sort_ascending = [False]

    @property
    def table(self):
        """
        :return: DataFrame
        """
        return self._df

    @property
    def table_organized(self):
        """
        :return: DataFrame
        """
        return self._organized_df

    def search(
            self,
            one_of,
            location="all",
            use_organized=True
    ):
        """
        Filters out rules that don't match search condition.
        E.g. if one_of=["Chisato", "Hina"], all rules with "Chisato" or "Hina" in antecedents or consequents
        will be returned.

        :param one_of: List; each element is search term, with entire list being a disjunction/OR
        :param location: "antecedents", "consequents", or "all"; where to look for search terms
        :param use_organized: Bool; whether to use organized table or not
        :return: DataFrame with results
        """
        if location not in ["all", "antecedents", "consequents"]:
            raise ValueError("invalid location argument: must be 'all', 'antecedents', or 'consequents'")

        if use_organized and self._organized_df is not None:
            rules = self._organized_df.copy()
        else:
            rules = self._df.copy()

        res = None
        filter_partials = []

        for term in one_of:

            # Do filtering/search of term at specified locations
            if location == "all":
                filter_partials.append(rules[rules["antecedents"].astype(str).str.contains(term)])
                filter_partials.append(rules[rules["consequents"].astype(str).str.contains(term)])
            else:
                filter_partials.append(rules[rules[location].astype(str).str.contains(term)])

            # Union partial filter results to get final result
            if res is not None:
                filter_partials.append(res)
            res = pd.concat(filter_partials).drop_duplicates()
            filter_partials = []

        # Resort with original sort order
        return res.sort_values(by=self._sort_by, ascending=self._sort_ascending)

    def organize(
            self,
            min_antecedents=1,
            min_consequents=1,
            min_rule_length=2,
            max_antecedents=None,
            max_consequents=None,
            max_rule_length=None,
            sort_by=None,
            sort_ascending=None,
    ):
        """
        Filter and sort own DataFrame table, with the intent of making data more readable.
        The new table is saved to its own attribute, while the original is retained.

        :param min_antecedents: Int; min to keep
        :param min_consequents: Int; min to keep
        :param min_rule_length: Int; minimum length of antecedents + consequents to keep
        :param max_antecedents: Int or None; max to keep
        :param max_consequents: Int or None; max to keep
        :param max_rule_length: Int or None; maximum length of antecedents + consequents to keep
        :param sort_by: List of Strings; column names
        :param sort_ascending: List of Bool; parallel to _sort_by and determines sort order of corresponding column
        """

        if sort_by is None:
            sort_by = ["confidence"]
        if sort_ascending is None:
            sort_ascending = ["False"]

        rules = self._df.copy()
        rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
        rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))
        rules["rule_len"] = rules.apply(lambda row: row.antecedent_len + row.consequent_len, axis=1)

        # Filter
        filtered = rules[
            (rules["antecedent_len"] >= min_antecedents) &
            (rules["consequent_len"] >= min_consequents) &
            (rules["rule_len"] >= min_rule_length)
            ]
        filtered = filtered[filtered["antecedent_len"] <= max_antecedents] if max_antecedents else filtered
        filtered = filtered[filtered["consequent_len"] <= max_consequents] if max_consequents else filtered
        filtered = filtered[filtered["rule_len"] <= max_rule_length] if max_rule_length else filtered

        # Sort
        self._organized_df = filtered.sort_values(by=sort_by, ascending=sort_ascending)
        self._sort_by = sort_by
        self._sort_ascending = sort_ascending
