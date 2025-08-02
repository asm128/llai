#include "llai_ranking.h"
#include "llai_log.h"

using std::string_view, std::vector, std::unordered_map, std::size, std::span;

static int32_t  test_query  (const llai::TokenWeightMap & idf_scores, const span<const llai::TokenWeightMap> weighted, const string_view queryToMatch) {
    vector<llai::TokenRange>    query_token_ranges;
    llai::TokenWeightMap        query_term_frequencies;   
    llai::TokenWeightLimits     query_freq_limits;        
    vector<string_view>         query_token_views; 
    llai::TokenWeightMap        query_weighted;
    llai::tokenize(queryToMatch, query_token_ranges);
    query_token_views.resize(query_token_ranges.size());
    llai::term_frequency(queryToMatch, query_token_ranges, query_token_views, query_term_frequencies, query_freq_limits);
    llai::weight_terms(query_term_frequencies, idf_scores, query_weighted); // Weight terms: TF * IDF
    int32_t                     bestMatch       = -1;
    double                      bestSimilarity  = 0;
    for(int iDoc = 0; iDoc < size(weighted); ++iDoc) {     // Cosine similarity between weighted vectors
        const double similarity = llai::cosine_similarity(query_weighted, weighted[iDoc]);
        if(bestSimilarity < similarity) {
            bestSimilarity = similarity;
            bestMatch      = iDoc;
		}
        //const auto & comparedDoc = docs[iDoc];
        //log_debug("similarity:%f. comparedDoc:'%.*s'.", (float)similarity, (int)comparedDoc.size(), comparedDoc.data());
    }
    //if(bestMatch < 0) {
    //    log_error("No match found for query: '%.*s'.", (int)queryToMatch.size(), queryToMatch.data());
    //    return -1; // No match found
    //}
    //const auto & comparedDoc = docs[bestMatch];
    //log_debug("Best match:\n"
    //    "similarity:%f. comparedDoc:'%.*s'.", (float)bestSimilarity, (int)comparedDoc.size(), comparedDoc.data());
    return bestMatch; // Return the index of the best matching document
}

int test_ranking(span<const string_view> docs, llai::TokenWeightMap & idf_scores, span<llai::TokenWeightMap> weighted) {
    using namespace llai;
    vector<vector<llai::TokenRange>>        tokenRanges;        tokenRanges     .resize(size(docs));
    llai::tokenize(docs, tokenRanges);    // Tokenize documents

    vector<llai::TokenWeightMap >           term_frequencies;   term_frequencies.resize(tokenRanges.size());
    vector<llai::TokenWeightLimits>         freq_limits;        freq_limits     .resize(tokenRanges.size());
    llai::term_frequency(docs, tokenRanges, term_frequencies, freq_limits);    // Get term frequencies

    vector<vector<string_view>>             token_views;        token_views.resize(tokenRanges.size());
    for(int iDoc = 0; iDoc < size(docs); ++iDoc) {
        auto & doc_token_views = token_views[iDoc];
        const auto & doc_token_ranges = tokenRanges[iDoc];
        doc_token_views.resize(doc_token_ranges.size());
        llai::token_views(docs[iDoc], doc_token_ranges, doc_token_views);
    }
    llai::TokenWeightMap                    doc_occurrences;
    llai::inverse_document_frequency(token_views, idf_scores, doc_occurrences); // Calculate IDF
    for(int iDoc = 0; iDoc < size(docs); ++iDoc)     
        llai::weight_terms(term_frequencies[iDoc], idf_scores, weighted[iDoc]); // Weight terms: TF * IDF

    for(int iDocA = 0; iDocA < size(docs) - 1; ++iDocA) {     
        const auto & weightedA  = weighted[iDocA];
        const auto & docA       = docs[iDocA];
        log_debug("Comparing query: '%.*s'", (int)docA.size(), docA.data());
        for(int iDocB = iDocA + 1; iDocB < size(docs); ++iDocB) { // Cosine similarity between weighted vectors
            const auto & weightedB  = weighted[iDocB];
            const auto & docB       = docs[iDocB];
            double similarity = llai::cosine_similarity(weightedA, weightedB);
            log_debug("Cosine similarity: %f. document. '%.*s'."
                , (float)similarity
                , (int)docB.size(), docB.data()
                );
        }
    }
    return 0;
}
int test_ranking_2(span<const string_view> docs, llai::TokenWeightMap & idf_scores, span<llai::TokenWeightMap> weighted) {
    using namespace llai;
	llai::load_docs(docs, idf_scores, weighted); // Load documents and calculate IDF and weighted terms
    for(int iDocA = 0; iDocA < size(docs) - 1; ++iDocA) {     
        const auto & weightedA  = weighted[iDocA];
        const auto & docA       = docs[iDocA];
        log_debug("Comparing query: '%.*s'", (int)docA.size(), docA.data());
        for(int iDocB = iDocA + 1; iDocB < size(docs); ++iDocB) { // Cosine similarity between weighted vectors
            const auto & weightedB  = weighted[iDocB];
            const auto & docB       = docs[iDocB];
            double similarity = llai::cosine_similarity(weightedA, weightedB);
            log_debug("Cosine similarity: %f. document. '%.*s'."
                , (float)similarity
                , (int)docB.size(), docB.data()
                );
        }
    }
    return 0;
}


int main() {
    string_view                docs[] = 
        { "apple banana apple orange"
        , "banana fruit apple banana"
        , "celular motorola con camara triple"
        , "iphone con chip bionic y buena camara"
        , "smartphone barato con android"
        , "celular con buena bateria y pantalla grande"
        , "The dog (Canis familiaris or Canis lupus familiaris) is a domesticated descendant of the gray wolf."
        , "This radio frequency link connects to the switching systems of a mobile phone operator, providing access to the public switched telephone network (PSTN)."
        , "The dog was the first species to be domesticated by humans, over 14,000 years ago and before the development of agriculture."
        , "These include text messaging, multimedia messaging, email, and internet access (via LTE, 5G NR or Wi-Fi), as well as short-range wireless technologies like Bluetooth, infrared, and ultra-wideband (UWB)."
        , "Dogs have been bred for desired behaviors, sensory capabilities, and physical attributes. Dog breeds vary widely in shape, size, and color." 
        , "In addition, they enable multimedia playback and streaming, including video content, as well as radio and television streaming."
        , "They have the same number of bones (with the exception of the tail), powerful jaws that house around 42 teeth, and well-developed senses of smell, hearing, and sight."
        , "Mobile phones also support a variety of multimedia capabilities, such as digital photography, video recording, and gaming."
        , "Compared to humans, dogs possess a superior sense of smell and hearing, but inferior visual acuity."
        , "Beyond traditional voice communication, digital mobile phones have evolved to support a wide range of additional services."
        , "Communication in dogs includes eye gaze, facial expression, vocalization, body posture (including movements of bodies and limbs), and gustatory communication (scents, pheromones, and taste)."
        , "In 2024, the top smartphone manufacturers worldwide were Samsung, Apple and Xiaomi; smartphone sales represented about 50 percent of total mobile phone sales.[6][7]"
        , "Dogs perform many roles for humans, such as hunting, herding, pulling loads, protection, companionship, therapy, aiding disabled people, and assisting police and the military."
        , "In 1983, the DynaTAC 8000x was the first commercially available handheld mobile phone."
        , "They mark their territories by urinating on them, which is more likely when entering a new environment."
        , "In 1979, Nippon Telegraph and Telephone (NTT) launched the world's first cellular network in Japan.[3]"
        , "Over the millennia, dogs have uniquely adapted to human behavior; this adaptation includes being able to understand and communicate with humans."
        , "The first handheld mobile phone was demonstrated by Martin Cooper of Motorola in New York City on 3 April 1973, using a handset weighing c. 2 kilograms (4.4 lbs).[2]"
        , "As such, the human-canine bond has been a topic of frequent study, and dogs' influence on human society has given them the sobriquet of c\"man's best friend\"."
        , "Today, mobile phones are globally ubiquitous,[11] and in almost half the world's countries, over 90%% of the population owns at least one.[12]"
        , "The dog is the most popular pet in the United States, present in 34-40%% of households. "
        , "A mobile phone or cell phone is a portable telephone that allows users to make and receive calls over a radio frequency link while moving within a designated telephone service area, unlike fixed-location phones (landline phones)." 
        , "Also called the domestic dog, it was selectively bred from a population of wolves during the Late Pleistocene by hunter-gatherers."
        , "Modern mobile telephony relies on a cellular network architecture, which is why mobile phones are often referred to as 'cell phones' in North America."
        , "Due to their long association with humans, dogs have gained the ability to thrive on a starch-rich diet that would be inadequate for other canids."
        , "Furthermore, mobile phones offer satellite-based services, such as navigation and messaging, as well as business applications and payment solutions (via scanning QR codes or near-field communication (NFC))."
        , "Mobile phones offering only basic features are often referred to as feature phones (slang: dumbphones), while those with advanced computing power are known as smartphones.[1]"
        , "The global dog population is estimated at 700 million to 1 billion, distributed around the world."
        , "From 1993 to 2024, worldwide mobile phone subscriptions grew to over 9.1 billion; enough to provide one for every person on Earth.[4][5]"
        , "For feature phones as of 2016, the top-selling brands were Samsung, Nokia and Alcatel.[8]"
        , "Developed countries make up approximately 20%% of the global dog population, while around 75%% of dogs are estimated to be from developing countries, mainly in the form of feral and community dogs."
        , "Mobile phones are considered an important human invention as they have been one of the most widely used and sold pieces of consumer technology.[9] The growth in popularity has been rapid in some places; for example, in the UK, the total number of mobile phones overtook the number of houses in 1999.[10]"
    };
    llai::TokenWeightMap                    idf_scores;
    vector<llai::TokenWeightMap>            weighted;           
    weighted.resize(size(docs));
	llai::load_docs(docs, idf_scores, weighted); // Load documents and calculate IDF and weighted terms
    //test_ranking(docs, idf_scores, weighted); 
    //test_ranking_2(docs, idf_scores, weighted); 
    {
        const string_view           text_query      = "un celu con muy buena onda";
        const int                   bestMatch       = test_query(idf_scores, weighted, text_query);
        if(bestMatch < 0) 
            log_error("No match found for query: '%.*s'.", (int)text_query.size(), text_query.data());
        else {
            const auto                  & comparedDoc   = docs[bestMatch];
            log_debug("Best match:'%.*s'.", (int)comparedDoc.size(), comparedDoc.data());
        }
    }
    {
        const string_view queries [] = 
            { "un celu con muy buena onda"
            , "Often in colloquial terms it is referred to as simply phone, mobile or cell."
            , "There are around 450 official dog breeds, the most of any mammal."
            , "A number of alternative words have also been used to describe a mobile phone, most of which have fallen out of use, including: \"mobile handset\", \"wireless phone\", \"mobile terminal\", \"cellular device\", \"hand phone\", and \"pocket phone\"."
            , "Present-day dogs are dispersed around the world."
            , "\"Mobile phone\" is the most common English language term, while the term \"cell phone\" is in more common use in North America[13] - both are in essence shorter versions of \"mobile telephone\" and \"cellular telephone\", respectively."
            , "The skull, body, and limb proportions between breeds display more phenotypic diversity than can be found within the entire order of carnivores."
            , "Most breeds were derived from small numbers of founders within the last 200 years."
            , "Their personality traits include hypersocial behavior, boldness, and aggression."
            , "These breeds possess distinct traits related to morphology, which include body size, skull shape, tail phenotype, fur type, and colour."
            , "Dogs began diversifying in the Victorian era, when humans took control of their natural selection."
            , "As such, humans have long used dogs for their desirable traits to complete or fulfill a certain work or role."
            , "Since then, dogs have undergone rapid phenotypic change and have been subjected to artificial selection by humans."
            , "Their behavioural traits include guarding, herding, hunting, retrieving, and scent detection."
            , "An example of this dispersal is the numerous modern breeds of European lineage during the Victorian era."
            };

        for(const auto & queryToMatch : queries) {
         	log_debug("\n---- Query to match: '%.*s'.", (int)queryToMatch.size(), queryToMatch.data());
            const int                   bestMatch       = test_query(idf_scores, weighted, queryToMatch);
            if(bestMatch < 0) 
                log_error("No match found for query: '%.*s'.", (int)queryToMatch.size(), queryToMatch.data());
            else {
                const auto                  & comparedDoc   = docs[bestMatch];
                log_debug("Best match:'%.*s'.", (int)comparedDoc.size(), comparedDoc.data());
            }
        }

    }
    return 0;
}