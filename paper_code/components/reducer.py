import pandas as pd
import numpy as np
import ast
import random
import wandb
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, SpectralClustering
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

from serve.utils_llm import get_llm_output, get_llm_embedding

systems_prompt = "Given a dataset of text outputs from two different large language models (LLMs), your task is to analyze and summarize the data based on specific characteristics. The goal is to identify and cluster similar behaviors or traits within the outputs, summarizing these into a concise list of commonly observed behaviors for each model. This analysis will help in understanding the general behaviors of these models for auditing, error discovery, and comparison purposes. Your outputs adhere to the format given by the user."
smaller_systems_prompt = (
    "You are a helpful assistant. Your outputs adhere to the format given by the user."
)


def cluster_dbscan(embeddings):
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(embeddings)
    # print out the size of each cluster
    print({i: np.sum(clustering.labels_ == i) for i in np.unique(clustering.labels_)})
    return clustering.labels_


def cluster_spectral(embeddings, n_clusters=5):
    clustering = SpectralClustering(
        n_clusters=n_clusters, assign_labels="discretize", random_state=0
    ).fit(embeddings)
    unique_labels = np.unique(clustering.labels_)
    print({i: np.sum(clustering.labels_ == i) for i in unique_labels})
    return clustering.labels_


def cluster_hierarchical(embeddings, n_clusters=5):
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(embeddings)
    unique_labels = np.unique(clustering.labels_)
    print({i: np.sum(clustering.labels_ == i) for i in unique_labels})
    return clustering.labels_


def cluster_kmeans(embeddings, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    print({i: np.sum(kmeans.labels_ == i) for i in range(n_clusters)})
    return kmeans.labels_


class Reducer:
    """Given a set of text, reduce the set to a smaller set of hypotheses"""

    def __init__(self, args):
        self.args = args
        random.seed(args.seed)

    @staticmethod
    def reduce_texts(texts, n_clusters=5, cluster_method="dbscan", kwargs={}):
        """
        Given a list of hypotheses, reduce them to a smaller set of hypotheses
        Return a list of reduced hypotheses along with a list of (text, parent_text) pairs and any tables we should be logging
        """
        return texts, [(t, t) for t in texts], []

    def reduce(self, hypotheses):
        return self.reduce_texts(
            hypotheses, n_clusters=self.args.k, cluster_method=self.args.cluster_method
        )


#####################################
### Clustering and Axis Reduction ###
#####################################


def get_clusters(
    texts, embedding_model, n_clusters=5, cluster_method="dbscan", kwargs={}
):
    """
    Clusters text into k clusters using hierarchical clustering
    Returns a dictionary of clusters where the key is the cluster number and the value is a list of texts in that cluster.
    """
    if embedding_model == "all-MiniLM-L6-v2":
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts)
    else:
        print(f"Using LLM embeddings ({embedding_model}) for clustering")

        def get_embedding(text):
            return get_llm_embedding(text, embedding_model)

        with ThreadPoolExecutor(max_workers=32) as executor:
            embeddings_list = list(executor.map(get_embedding, texts))

        embeddings = np.stack(embeddings_list)

    clusters = cluster_hierarchical(embeddings, n_clusters=n_clusters)

    # Group axes by cluster
    grouped_axes = {i: [] for i in range(n_clusters)}
    for axis, cluster in zip(texts, clusters):
        grouped_axes[cluster].append(axis)
    return grouped_axes


def get_cluster_axes(cluster, model="gpt-4", batch=100):
    cluster_axes_descriptions_prompt = [
        """The following are the axes of variation that you can consider when comparing the two model outputs along with a description of how two models (A and B) vary along that axis. Each axis has a name as well as a description of what it means to be low and high on this axis. Many of these axes of variations could be named incorrectly or redundant with other axes. I want to cluster these axes so that I can better understand the general patterns seen in these models without having to look through so many axes. Please cluster this large list of axes into a minimal set of parent axes that cover the entire axis list. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Please ensure that the descriptions of what makes a piece of text low or high on this axis are distinct and mutually exclusive such that given any pair of text outputs, a human could easily and reliably determine which model is higher or lower on that axis. If an axis applies to a specific type of task (e.g. coding), please ensure that the axis is named in a way that makes it clear what type of task it applies to.
                        
    Here are the axes of varaiation (note each axis is formatted {{axis name}}: High: {{high description}} Low: {{low description}}):
    {axes}

    Again I want to cluster these axes into a minimal set of parent axes that cover the entire axis list. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Your new set of axes should be distinct so each of the above axes fit under exactly one of your new axes. Please ensure each axis and parent axis contains an axis name and descriptions of what it means to score high or low on that axis in the same format as the provided axes.  Please ensure the descriptions of what is considered high and low on each axis is clear, concise, under 10 words.""",
        """I have a list of axes that I would like to convert into a list that I can parse in python. Here are the axes:
    {axes}

    Please structure your response as a list which can be parsed with ast.literal_eval() in Python. The format should be as follows:

    ["{{axis name}}:  High: {{new axis high description}} Low: {{new axis low description}}", ...]""",
    ]

    cluster_axes_descriptions_prompt_new = """The following are the axes of variation that you can consider when comparing the two model outputs along with a description of how two models (A and B) vary along that axis. Each axis has a name as well as a description of what it means to be low and high on this axis. Many of these axes of variations could be named incorrectly or redundant with other axes. I want to cluster these axes so that I can better understand the general patterns seen in these models without having to look through so many axes. Please cluster this large list of axes into a minimal set of parent axes that cover the entire axis list. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Your new set of axes should be distinct so each of the above axes fit under exactly one of your new axes. Please ensure that the descriptions of what makes a piece of text low or high on this axis are distinct and mutually exclusive such that given any pair of text outputs, a human could easily and reliably determine which model is higher or lower on that axis. If an axis applies to a specific type of task (e.g. coding), please ensure that the axis is named in a way that makes it clear what type of task it applies to.
                        
    Here are the axes of varaiation (note each axis is formatted {{axis name}}: High: {{high description}} Low: {{low description}}):
    {axes}

    Again I want to cluster these axes into a minimal set of parent axes that cover the entire axis list. Please ensure these parent axes' descriptions of what makes an item high or low on that axis align with the high and low descriptions of the axes they cover. Your new set of axes should be distinct so each of the above axes fit under exactly one of your new axes. Please ensure each axis and parent axis contains an axis name and descriptions of what it means to score high or low on that axis in the same format as the provided axes.  Please ensure the descriptions of what is considered high and low on each axis is clear, concise, under 10 words."""
    smaller_systems_prompt = "You are a helpful assistant. Your outputs adhere to the format given by the user."
    cluster = set(cluster)
    cluster_batch = random.sample(cluster, min(batch, len(cluster)))
    cluster_batch = sorted(cluster_batch)
    prompt_1 = cluster_axes_descriptions_prompt_new.format(
        axes="\n".join(cluster_batch)
    )
    cluster_1_reduced_axes = get_llm_output(
        prompt_1, model=model, system_prompt=smaller_systems_prompt
    )
    cluster_1_reduced_axes = cluster_1_reduced_axes.replace("*", "")

    cache = True
    for _ in range(3):
        try:
            history = [
                {"role": "user", "content": prompt_1},
                {"role": "assistant", "content": cluster_1_reduced_axes},
            ]
            prompt_2 = cluster_axes_descriptions_prompt[1].format(
                axes="\n".join(cluster_batch)
            )
            # cluster_1_reduced_axes_categorized = get_llm_output(prompt_2, model=model, system_prompt=smaller_systems_prompt,history=history, cache=cache)
            cluster_1_reduced_axes_categorized = get_llm_output(
                prompt_2, model=model, system_prompt=smaller_systems_prompt, cache=cache
            )
            # cut any thing before the [ and after the ]
            cluster_1_reduced_axes_categorized = cluster_1_reduced_axes_categorized[
                cluster_1_reduced_axes_categorized.find(
                    "["
                ) : cluster_1_reduced_axes_categorized.rfind("]")
                + 1
            ]
            print(f"cluster reduced {cluster_1_reduced_axes}")
            cluster_1_reduced_axes = ast.literal_eval(
                cluster_1_reduced_axes_categorized
            )
        except:
            print(f"Error parsing cluster axes {cluster_1_reduced_axes}")
            cache = False

    return prompt_1, cluster_1_reduced_axes


def match_axis_to_subaxis(axes, parent_axes, embedding_model):
    if embedding_model == "all-MiniLM-L6-v2":
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Generate embeddings
        axes_embeddings = model.encode(axes)
        parent_axes_embeddings = model.encode(parent_axes)
    else:
        print(f"Using LLM embeddings ({embedding_model}) to match axes to parent axes")

        def get_embedding(text):
            return get_llm_embedding(text, embedding_model)

        with ThreadPoolExecutor(max_workers=32) as executor:
            axes_embeddings_list = list(executor.map(get_embedding, axes))
            parent_axes_embeddings_list = list(executor.map(get_embedding, parent_axes))

        axes_embeddings = np.stack(axes_embeddings_list)
        parent_axes_embeddings = np.stack(parent_axes_embeddings_list)

    # Function to find the closest parent axis for each axis
    def find_closest_parent(axes_embeddings, parent_axes_embeddings):
        similarity_matrix = cosine_similarity(axes_embeddings, parent_axes_embeddings)
        closest_parent_indices = np.argmax(similarity_matrix, axis=1)
        return [parent_axes[index] for index in closest_parent_indices]

    # Categorize each axis
    categorized_axes = find_closest_parent(axes_embeddings, parent_axes_embeddings)
    return categorized_axes


def parse_reduced_axes_strings(input_string, num_final=10):
    result = []
    lines = input_string.split("\n")
    inside_list = False

    for line in lines:
        line = line.strip()
        # change to be num_final instead of 10
        # if line.startswith(set([f"{i}." for i in range(1, num_final + 1)])):
        if line.startswith(
            (
                "1.",
                "2.",
                "3.",
                "4.",
                "5.",
                "6.",
                "7.",
                "8.",
                "9.",
                "10.",
                "11.",
                "12.",
                "13.",
                "14.",
                "15.",
                "16.",
                "17.",
                "18.",
                "19.",
                "20.",
                "21.",
                "22.",
                "23.",
                "24.",
                "25.",
                "26.",
                "27.",
                "28.",
                "29.",
                "30.",
                "31.",
                "32.",
                "33.",
                "34.",
                "35.",
                "36.",
                "37.",
                "38.",
                "39.",
                "40.",
                "41.",
                "42.",
                "43.",
                "44.",
                "45.",
                "46.",
                "47.",
                "48.",
                "49.",
                "50.",
            )
        ):
            # Remove the leading number and dot
            line = line[line.index(".") + 1 :].strip()
            result.append(line)
        elif inside_list and not line:
            # Empty line after the list, exit the loop
            break

    return result


def simplify_axes(parent_axes, num_final):

    remove_duplicates = """Below is a list of axes with a description of what makes a piece of text low or high on this axis. Are there any axes that have similar meanings based off their low and high descriptions? Are there any sets of axes that would convey the same information to a user (e.g. level of detail)? Could any of the low and high descriptions be simplified to make them easier to understand?
    
Please remove any axes with roughly the same meaning and simplify the descriptions of what makes a piece of text low or high on this axis. Please ensure that the descriptions of what makes a piece of text low or high on this axis are distinct, useful, and mutually exclusive. Given any piece of text, a human should be able to easily and reliably determine if this text falls high or low on each axis. 

Here is the list of axes:
{axes}

Please return the simplified list of axes and the descriptions of what makes a piece of text low or high on this axis. These axes should contain only one concept and should be human interpretable. 
Some examples of bad axes include:
- "Configuration Clarity: High: Clearly defined structure and purpose. Low: Vaguely defined, minimal purpose." -> This axes is bad because it is not clear what a clearly defined purpose means nor what a vaugely defined purpose means. 
- "Language and Communication: High: Varied/precise, complex structure. Low: Straightforward, simple or general language." -> This axes is bad because it combines multiple concepts into one axis.
- "Content Quality: High: High quality, engaging, informative. Low: Low quality, unengaging, uninformative." -> This axes is bad because it is not clear what high quality means nor what low quality means.

    Some examples of good axes include:
- "Complexity: High: Complex, multi-layered, intricate. Low: Simple, straightforward, easy to understand."
- "Efficiency (coding): High: Code optimized for runtime, minimal memory usage. Low: Code inefficient, high memory usage."

Some examples of axes which should be combined include:
- "Emotional Tone: High: Contains emotionally charged language. Low: Maintains a neutral tone." and "Empathy: High: Shows empathy. Low: Only factual answers without empathy." are redundant because they both measure the emotional content of the text. If two similar axes are found, keep the one that is more informative or more specific.

Please maintain the format of the original axes and return a list like ["{{axis_name}}: High: {{high description}} Low: {{low description}}", ...]. I should be able to parse this output into a string using ast.literal_eval. If the original list does not contain any redundant axes, please return the original list."""

    conversion_prompt = """I have an LLM output which list of axes that I would like to format. Here are the axes:
    
{axes}

My goal is to convert this into a list that I can parse with  with ast.literal_eval() in python. The format should be as follows:
["{{axis name}}:  High: {{new axis high description}} Low: {{new axis low description}}", ...]"""

    reduce_to_10 = """Below is a list of axes with a description of what makes a piece of text low or high on this axis. I would like to summarize this list to at most {number} representative axes.

Here is the list of axes:
{axes}

These axes should contain only one concept and should be human interpretable. Some examples of bad axes include:
- "Configuration Clarity: High: Clearly defined structure and purpose. Low: Vaguely defined, minimal purpose." -> This axis is bad because it is not clear what a clearly defined purpose means nor what a vaguely defined purpose means. 
- "Language and Communication: High: Varied/precise, complex structure. Low: Straightforward, simple or general language." -> This axis is bad because it combines multiple concepts into one axis.
- "Content Quality: High: High quality, engaging, informative. Low: Low quality, unengaging, uninformative." -> This axis is bad because it is not clear what high quality means nor what low quality means.

Some examples of good axes include:
- "Complexity: High: Complex, multi-layered, intricate. Low: Simple, straightforward, easy to understand."
- "Efficiency (coding): High: Code optimized for runtime, minimal memory usage. Low: Code inefficient, high memory usage."

Some examples of axes which should be combined include:
- "Emotional Tone: High: Contains emotionally charged language. Low: Maintains a neutral tone." and "Empathy: High: Shows empathy. Low: Only factual answers without empathy." are redundant because they both measure the emotional content of the text. If two similar axes are found, keep the one that is more informative or more specific.

Please return the simplified list of <={number} axes with any redundant axes removed and the descriptions of what makes a piece of text low or high on this axis simplified. Are there any axes which convey roughly the same information? Are there any axes where almost all samples which score highly on one axis would also score highly on the other? 

Please maintain the format of the original axes and return a numbered list. Each element should be structured as follows:
"{{axis_name}}: High: {{high description}} Low: {{low description}}" """

    prompt = remove_duplicates.format(axes="\n".join(parent_axes))
    old_parent_axes = parent_axes
    cache = True
    # for _ in range(3):
    try:
        response = get_llm_output(
            prompt, model="gpt-4o", system_prompt=smaller_systems_prompt, cache=cache
        )
        parent_axes = ast.literal_eval(
            response[response.find("[") : response.rfind("]") + 1]
        )
    except:
        cache = False
        response = get_llm_output(
            conversion_prompt.format(response),
            model="gpt-4o-mini",
            system_prompt=smaller_systems_prompt,
            cache=cache,
        )
        print(f"Error parsing axes simplification\n{response}")

    cache = True
    for _ in range(3):
        if len(parent_axes) > num_final:
            prompt = reduce_to_10.format(number=num_final, axes="\n".join(parent_axes))
            try:
                response = get_llm_output(
                    prompt,
                    model="gpt-4o",
                    system_prompt=smaller_systems_prompt,
                    cache=cache,
                )
                parent_axes = parse_reduced_axes_strings(response, num_final)
            except:
                print(f"Error parsing axes simplificationx2\n{response}")
                cache = False

    return [p.replace("*", "") for p in parent_axes]


class AxisReducer(Reducer):

    def __init__(self, args):
        super().__init__(args)
        random.seed(args.seed)

    def reduce_texts(
        self,
        texts,
        embedding_model="all-MiniLM-L6-v2",
        summarization_model="gpt-4",
        n_clusters=5,
        cluster_method="dbscan",
        kwargs={},
    ):
        """
        Given a list of hypotheses, reduce them to a smaller set of hypotheses
        Return a list of reduced hypotheses along with a list of (text, parent_text) pairs
        """

        if len(texts) < 100:
            print(
                f"Skipping clustering because the number of axes is less than 100 (length = {len(texts)})"
            )
            # Skip clustering
            axes = texts
            # Process all axes together
            prompt_1, parent_axes = get_cluster_axes(sorted(axes), model="gpt-4o")
            llm_logs = {0: {"prompt_1": prompt_1, "output_1": parent_axes}}
            df_cluster = {"subaxis": [], "cluster": []}
            cluster = 0
            for axis in parent_axes:
                df_cluster["subaxis"].append(axis)
                df_cluster["cluster"].append(cluster + 1)
            # Create DataFrame
            df_cluster = pd.DataFrame(df_cluster)
            print(f"Processed all axes without clustering (length = {len(axes)}):")
            llm_outputs = pd.DataFrame(llm_logs).T
        else:
            print(
                f"Proceeding with clustering because the number of axes is greater than 100 (length = {len(texts)})"
            )
            # Proceed with clustering
            grouped_axes = get_clusters(
                texts,
                embedding_model,
                n_clusters=n_clusters,
                cluster_method=cluster_method,
            )
            all_df_cluster, llm_logs = [], {}
            for cluster, axes in grouped_axes.items():
                prompt_1, parent_axes = get_cluster_axes(sorted(axes), model="gpt-4o")
                llm_logs[cluster] = {"prompt_1": prompt_1, "output_1": parent_axes}
                df_cluster = {"subaxis": [], "cluster": []}
                for axis in parent_axes:
                    df_cluster["subaxis"].append(axis)
                    df_cluster["cluster"].append(cluster + 1)
                # all_cluster_axes.append(cluster_axes)
                all_df_cluster.append(pd.DataFrame(df_cluster))
                print(
                    f"Cluster {cluster + 1} (length = {len(axes)}) (df length = {len(df_cluster)}):"
                )
                print("")  # New line for readability between clusters
            llm_outputs = pd.DataFrame(llm_logs).T
            df_cluster = pd.concat(all_df_cluster)

        llm_outputs = pd.DataFrame(llm_logs).T
        print(df_cluster)
        parent_axes = df_cluster["subaxis"].unique()
        print("Parent Axes before simplification:", parent_axes)
        parent_axes = [t for t in parent_axes if "Axis" not in t]
        parent_axes = simplify_axes(parent_axes, num_final=self.args.num_axes_generated)
        print("\n\nParent axes AFTER simplification:", parent_axes)
        # match each axis to its parent
        categorized_axes = match_axis_to_subaxis(texts, parent_axes, embedding_model)
        df_cluster["axis"] = match_axis_to_subaxis(
            list(df_cluster["subaxis"]), parent_axes, embedding_model
        )
        # sort categories by frequency
        ordered_parent_axes = df_cluster["axis"].value_counts().index.tolist()
        wandb.summary["num_axes"] = len(texts)
        wandb.summary["num_parent_axes"] = len(parent_axes)

        return (
            ordered_parent_axes,
            categorized_axes,
            {"df_cluster": df_cluster, "llm_outputs": llm_outputs},
        )

    def reduce(self, hypotheses):
        return self.reduce_texts(
            hypotheses,
            n_clusters=self.args.k,
            cluster_method=self.args.cluster_method,
            embedding_model=self.args.embedding_model,
        )
