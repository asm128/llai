#include "llai_ranking.h"
#include "llai_log.h"

#include <unordered_set>
#include <cctype>

using std::bad_alloc;
using std::span, std::string_view, std::vector, std::unordered_map, std::unordered_set;
using std::min, std::max, std::size, std::isspace;

#define log_rank_debug(fmt, ...)	do {} while(0) // log_debug("|ranking|" fmt, __VA_ARGS__) //

#define log_token_debug(fmt, ...)	log_debug("|token|" fmt, __VA_ARGS__) //

static bool is_terminator       (const string_view & text, const string_view & terminator) {
	if(text.size() < terminator.size()) 
        return false;
    return terminator.size() && terminator == text.substr(0, terminator.size());
}

int32_t llai::inverse_document_frequency(const span<vector<string_view>>& documents, TokenWeightMap & idf_scores, TokenWeightMap & document_occurrences) {
    for (const auto & doc : documents) {
        unordered_set<string_view> unique_terms;
        for (auto term : doc) 
            unique_terms.insert(term);
        for (auto term : unique_terms) {
            document_occurrences[term] += 1;
			log_rank_debug("Document term: '%.*s' occurs %u times.", (int)term.size(), term.data(), (uint32_t)document_occurrences[term]);
        }
    }
    const double total_documents = (double)documents.size();
    for (const auto & [term, count] : document_occurrences)
        idf_scores[term] = log(total_documents / (1.0 + count));
    return (int32_t)idf_scores.size();
}
int32_t llai::weight_terms
    ( const TokenWeightMap & term_freq_map
    , const TokenWeightMap & inverse_doc_freq_map
    , TokenWeightMap & weighted_terms
    ) {
    for (const auto & [term, tf_value] : term_freq_map) {
        auto it = inverse_doc_freq_map.find(term);
        if (it == inverse_doc_freq_map.end()) 
            log_rank_debug("Term not found: '%.*s'.", (int)term.size(), term.data());
        else {
            weighted_terms[term] = tf_value * it->second;        
		    log_rank_debug("Weighted term: '%.*s' = %f.", (int)term.size(), term.data(), weighted_terms[term]);
        }
    }
    return (int32_t)weighted_terms.size();
}
double llai::cosine_similarity
    ( const TokenWeightMap & tf_idf_1
    , const TokenWeightMap & tf_idf_2
    , double epsilon
    ) {
    double dot = 0, norm1 = 0, norm2 = 0;
    for (const auto & [k, v] : tf_idf_1) {
        auto it = tf_idf_2.find(k);
        if (it != tf_idf_2.end()) 
            dot += v * it->second;
        norm1 += v * v;
    }
    for (const auto & [_, v] : tf_idf_2) 
        norm2 += v * v;
    return dot / (sqrt(norm1) * sqrt(norm2) + epsilon);
}
int32_t llai::token_views(const string_view & document, const span<const TokenRange> & tokenRanges, span<string_view> views) {
    for(int iToken = 0; iToken < size(tokenRanges); ++iToken) {
        const auto & tokenRange = tokenRanges[iToken];
        views[iToken] = document.substr(tokenRange.Offset, tokenRange.Size);       // Prepare docs as vectors of tokens (string_views)
    }
    return 0;
}
int32_t llai::term_frequency    (const span<const string_view> & documents, const span<vector<TokenRange>> tokenRanges, span<TokenWeightMap> frequencies, span<TokenWeightLimits> limits) {
    for(int iDoc = 0; iDoc < size(documents); ++iDoc)
        llai::term_frequency(documents[iDoc], tokenRanges[iDoc], frequencies[iDoc], limits[iDoc]);    // Get term frequencies
    return 0;
}
int32_t llai::term_frequency    (const string_view & text, const span<const TokenRange> & tokenRanges, TokenWeightMap & frequency_map, TokenWeightLimits & frequency_limits) {
    frequency_limits = {};
    for (auto tokenRange : tokenRanges) 
        frequency_map[text.substr(tokenRange.Offset, tokenRange.Size)] += 1.0f;
    for (auto & [token, count] : frequency_map) {
        count /= tokenRanges.size();
        if(count < frequency_limits.Min.second || 0 == frequency_limits.Min.second) 
            frequency_limits.Min = {token, count};
        if(count > frequency_limits.Max.second) 
            frequency_limits.Max = {token, count};
		//log_rank_debug("token '%.*s' weights %f.", (int)token.size(), token.data(), (float)count);
    }
    log_rank_debug("frequency_limits: min '%.*s' = %f, max '%.*s' = %f."
        , (int)frequency_limits.Min.first.size(), frequency_limits.Min.first.data(), frequency_limits.Min.second
		, (int)frequency_limits.Max.first.size(), frequency_limits.Max.first.data(), frequency_limits.Max.second
        );
    return (int32_t)tokenRanges.size();
}
int32_t llai::term_frequency    (const span<const string_view> & documents, const span<vector<TokenRange>> tokenRanges, span<vector<string_view>> tokenViews, span<TokenWeightMap> frequencies, span<TokenWeightLimits> limits) {
    for(int iDoc = 0; iDoc < size(documents); ++iDoc) {
        tokenViews[iDoc].resize(tokenRanges[iDoc].size());
        llai::term_frequency(documents[iDoc], tokenRanges[iDoc], tokenViews[iDoc], frequencies[iDoc], limits[iDoc]);    // Get term frequencies
    }
    return 0;
}
int32_t llai::term_frequency    (const string_view & text, const span<const TokenRange> & tokenRanges, span<string_view> tokenViews, TokenWeightMap & frequency_map, TokenWeightLimits & frequency_limits) {
    frequency_limits = {};
    for(int iToken = 0; iToken < size(tokenRanges); ++iToken) {
        const auto tokenRange = tokenRanges[iToken];
        const auto & tokenView = tokenViews[iToken] = text.substr(tokenRange.Offset, tokenRange.Size);
        frequency_map[tokenView] += 1.0f;
    }
    for (auto & [token, count] : frequency_map) {
        count /= tokenRanges.size();
        if(count < frequency_limits.Min.second || 0 == frequency_limits.Min.second) 
            frequency_limits.Min = {token, count};
        if(count > frequency_limits.Max.second) 
            frequency_limits.Max = {token, count};
		//log_rank_debug("token '%.*s' weights %f.", (int)token.size(), token.data(), (float)count);
    }
    log_rank_debug("frequency_limits: min '%.*s' = %f, max '%.*s' = %f."
        , (int)frequency_limits.Min.first.size(), frequency_limits.Min.first.data(), frequency_limits.Min.second
		, (int)frequency_limits.Max.first.size(), frequency_limits.Max.first.data(), frequency_limits.Max.second
        );
    return (int32_t)tokenRanges.size();
}

int32_t llai::term_frequency    (const span<const string_view> & tokenViews, TokenWeightMap & frequency_map, TokenWeightLimits & frequency_limits) {
    frequency_limits = {};
    for (auto tokenView : tokenViews) 
        frequency_map[tokenView] += 1.0f;
    for (auto & [token, count] : frequency_map) {
        count /= tokenViews.size();
        if(count < frequency_limits.Min.second || 0 == frequency_limits.Min.second) 
            frequency_limits.Min = {token, count};
        if(count > frequency_limits.Max.second) 
            frequency_limits.Max = {token, count};
		log_rank_debug("token '%.*s' weights %f.", (int)token.size(), token.data(), count);
    }
    log_rank_debug("frequency_limits: min '%.*s' = %f, max '%.*s' = %f."
        , (int)frequency_limits.Min.first.size(), frequency_limits.Min.first.data(), frequency_limits.Min.second
		, (int)frequency_limits.Max.first.size(), frequency_limits.Max.first.data(), frequency_limits.Max.second
        );
    return (int32_t)tokenViews.size();
}
int32_t llai::tokenize          (const span<const string_view> & documents, span<vector<TokenRange>> tokenRanges, const string_view & terminator) {
    for(int iDoc = 0; iDoc < size(documents); ++iDoc) {
        const auto & document = documents[iDoc];
        if(0 > llai::tokenize(document, tokenRanges[iDoc], terminator)) {
            log_error("Failed to tokenize document at %i: '%.*s'.", iDoc, (int)document.size(), document.data());
            return (1 + iDoc) * -1;
        }
        log_rank_debug("Tokenized document at %i: '%.*s'.", iDoc, (int)document.size(), document.data());   
    }
    return (int32_t)documents.size();
}

static inline bool is_skippable(char c) { return isspace(c) || c < 'A' || c > 'z'; } 

int32_t llai::tokenize          (const string_view & text, vector<TokenRange> & tokenRanges, const string_view & terminator) {
    uint32_t start = 0;
    while(start < text.size()) {
        while(start < text.size() && is_skippable(text[start]) && not is_terminator(text.substr(start), terminator))
            ++start;
        uint32_t end = start;
		while(end < text.size() && not is_skippable(text[end]) && not is_terminator(text.substr(end), terminator))
            ++end;
        if (start < end) {
            try { 
                tokenRanges.push_back({start, end - start}); 
				log_token_debug("Token: '%.*s' at [%u, %u]", (int)(end - start), &text[start], start, end);
            }
            catch (const bad_alloc & e) {
				log_error("exception message:'%s'", e.what());
                return -1;
            }
        }
        start = end;
    }
	return start; // Return the position after the last token
}

int32_t llai::load_docs(const span<const string_view> & docs, llai::TokenWeightMap & idf_scores, span<llai::TokenWeightMap> weighted) {
    vector<vector<llai::TokenRange>>        tokenRanges;        tokenRanges     .resize(size(docs));
    llai::tokenize(docs, tokenRanges);    // Tokenize documents

    vector<llai::TokenWeightMap >           term_frequencies;   term_frequencies.resize(tokenRanges.size());
    vector<llai::TokenWeightLimits>         freq_limits;        freq_limits     .resize(tokenRanges.size());
    vector<vector<string_view>>             token_views;        token_views     .resize(tokenRanges.size());
    llai::term_frequency(docs, tokenRanges, token_views, term_frequencies, freq_limits);    // Get term frequencies

    llai::TokenWeightMap                    doc_occurrences;
    llai::inverse_document_frequency(token_views, idf_scores, doc_occurrences); // Calculate IDF
    for(int iDoc = 0; iDoc < size(docs); ++iDoc)     
        llai::weight_terms(term_frequencies[iDoc], idf_scores, weighted[iDoc]); // Weight terms: TF * IDF
    return 0;
}
