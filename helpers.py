import matplotlib.pyplot as plt

class Helpers():

    def plot_graph(self, feature_generator, n):
        term_counts = {}

        for term_count in feature_generator.term_counts:
            for key, value in term_count.items():
                term_counts[str(key)] = term_counts.get(str(key), 0) + value

            top_terms = dict(sorted(term_counts.items(), key=lambda item: item[1], reverse=True)[:n])

        plt.figure(figsize=(10, 6))
        plt.bar(top_terms.keys(), top_terms.values(), color='skyblue')
        plt.xlabel('Terms')
        plt.ylabel('Counts')
        plt.title('Terms and Counts Bar Chart')
        plt.xticks(rotation=45, ha='right')
        plt.show()
