# implementation of Kozlowski et al. (2019)
# input files can be created by following the tutorial from using the following Github page
# https://gitlab.uzh.ch/noah.bubenhofer/kodup-germanistik/-/tree/master/4._Korpusanalyse/Word_Embeddings 

# input files can also be found in the folder 'model' and has the format .model
# output will be a visualization of cosine similarity

import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

# This block calculates the cosine similarity between a word vector and a semantic dimension.
# input: list of words to analyze
def we_basics(infile):
    model = Word2Vec.load(infile)
    # A word list can be entered as a string 'Maske' or as a tuple ('Maske', 'FFP2') for very similar words.
    # Note: For the Telegram model, the character 'ß' must be used (for the KD model, 'ß' can be retained).
    word_list = [
        ('Maskenpflicht', 'Gesichtsmaske', 'Maske', 'FFP2'),
        ('Quarantäne', 'Selbstquarantäne', 'Heimquarantäne', 'Quarantänefrist', 'Zwangsquarantäne'),
        ('Isolation', 'Selbstisolation'),
        ('Homeoffice', 'Home-Schooling', 'Home_Office', 'Heimbüro', 'Homeofficepflicht', 'Homeoffice-Pflicht', 'Videokonferenzen'),
        ('Zertifikatspflicht', '3G-Regeln', '3-G-Regeln'),
        ('2G+', '2G', '2G-plus', '2G-Pflicht'),
        ('Zugangsbeschränkung', 'maximal', 'Beschränkung', 'begrenzt', 'Besucherlimite', 'Personenzahl', 'Kapazitätsbeschränkungen'),
        ('Menschenansammlung', 'Menschenmasse'),
        ('Öffnung'),
        ('Aufhebung', 'Lockerung'),
        ('Einreisebeschränkung', 'Grenzkontrolle', 'Risikoland', 'eingereiste'),
        ('Distanz', 'Abstand'),
        ('Kontaktdaten', 'Contact-Tracing', 'Kontakt', 'Kontaktperson'),
        ('Hygiene', 'desinfizieren', 'Desinfektionsmittel'),
        ('Spitalbetten', 'Spitalpersonal', 'Intensivpflegeplätzen', 'Überlastung', 'Intensiv-Betten', 'Intensivbetten'),
        ('Coronatest', 'Selbsttest', 'Testergebnis', 'Virustest', 'Nasenabstrich', 'PCR-Test', 'Antigen-Test', 'Schnelltest'),
        ('Impfung', 'Spritze', 'Impfdosis', 'Erstimpfung', 'Corona-Impfung', 'Coronaimpfung', 'impfen', 'Impfschutz'),
        ('Booster', 'Boosterimpfung', 'Drittimpfung', 'Auffrischimpfung'),
        # works only for Telegram Data
        # ('natürlich_Immunisierung')
    ]
    # Ideally, there should be up to 42 antonym pairs. However, it also works with fewer antonym pairs. The order of antonyms does not matter.
    # Note: For the Telegram model, the character 'ß' must be used (for the KD model, 'ß' can be retained).
   

    antonym_list = [('Solidarität', 'Egoismus'),
('Zusammenhalt', 'Eigenverantwortung'),
('Zusammenstehen', 'Isolierung'),
('Zusammenhalt', 'Ausschluß'),
('Gemeinschaftssinn', 'Individualismus'),
('Hilfsbereitschaft', 'Opportunismus'),
('Opferbereitschaft', 'Eigennutz'),
('Mitmenschlichkeit', 'Apathie'),
('Mitgefühl', 'Gleichgültigkeit'),
('Dankbarkeit', 'Undankbarkeit'),
('Gemeinsinn', 'Eigensinn'),
('Verantwortung', 'Verantwortungslosigkeit'),
('Anteilnahme', 'Gleichgültigkeit'),
('Verantwortung', 'Selbstverantwortung'),
('Gesellschaft', 'Selbstbestimmtheit'),
('Gemeinnützigkeit', 'Eigeninteresse'),
('Allgemeinwohl', 'Eigenliebe'),
('Zusammengehörigkeit', 'Selbstsucht'),
('Empathie', 'Indifferenz'),
('Selbstlosigkeit', 'Herzlosigkeit'),
('Verantwortungsgefühl', 'Empathielosigkeit'),
('Solidarität', 'Teilnahmslosigkeit'),
('Betroffenheit', 'Intoleranz'),
('Gemeinwohl', 'Egoismus'),
('gemeinnützig', 'eigennützig'),
('solidarisch', 'unsolidarisch'),
('sozial', 'asozial'),
('altruistisch', 'egoistisch'),
('hilfsbereit', 'selbstbestimmt'),
('opferbereit', 'selbstsüchtig'),
('selbstlos', 'egozentrisch'),
('grossherzig', 'herzlos'),
('kompromissbereit', 'rücksichtslos'),
('wohltätig', 'opportunistisch'),
('empathisch', 'empathielos ')]
    vis_list = []
    cultural_dim = 0
    synonyms = 0
    # semantic dimension is calculated. Tuple [0] - Tuple [1]
    for key, value in antonym_list:
        try:
            x = model.wv[key] - model.wv[value]
            cultural_dim += x
        except KeyError:
            word_list = [word.replace('ß', 'ss') if isinstance(word, str) else (word[0].replace('ß', 'ss'), word[1].replace('ß', 'ss')) for word in word_list]

    # divide by length of antonym for the average
    with_div = cultural_dim / len(antonym_list)
    #reshape, in order to calculate cosine similarity 1 | -1 
    vector_cultural = with_div.reshape(1, -1)
    
    # compare each measure with the semantic dimension
    for word_input in word_list:
        # identify the data type
        if type(word_input) is tuple:
            for word_syn in word_input:
                synonym = (model.wv[word_syn])
                synonyms += synonym
            vector_word = (synonyms / len(synonyms)).reshape(1, -1)
            similarity = cosine_similarity(vector_cultural, vector_word)
            thistuple = tuple((similarity[0][0], word_input[0]))
            vis_list.append(thistuple)
        
        else:
            word = (model.wv[word_input])
            vector_word = word.reshape(1, -1)
            similarity = cosine_similarity(vector_cultural, vector_word)
            thistuple = tuple((similarity[0][0], word_input))
            vis_list.append(thistuple)
    vis_list_sort = sorted(vis_list)
    print(vis_list_sort)
    return vis_list_sort

# visualization of cosine similarity
def plot_cosine_similarity(cos_similarities_1):
    fig = plt.figure(figsize=(6, 3))

    # Create the first polar plot
    ax1 = fig.add_subplot(121, projection='polar')
    for cos_sim, word in cos_similarities_1:
        # Map cosine similarity to 180-degree scale
        mapped_value = ((cos_sim + 1) / 2) * 180
        # Plot the mapping on the 180-degree scale
        ax1.plot([0, mapped_value * np.pi / 180], [0, 1], label=word + ' ' + str(cos_sim))

    ax1.set_thetamin(0)
    ax1.set_thetamax(180)
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location('W')
    ax1.legend(bbox_to_anchor=(2, 0), loc='center', borderaxespad=0)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculation with word embeddings")
    parser.add_argument("in_file", type=str, help="input data: word vector format file")
    args = parser.parse_args()

    words = we_basics(args.in_file)
    plot_cosine_similarity(words)