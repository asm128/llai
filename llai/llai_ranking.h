#include <vector>
#include <string_view>
#include <unordered_map>
#include <span>
//#include <algorithm>

#ifndef LLAI_RANKING_H
#define LLAI_RANKING_H

namespace llai
{
    typedef std::unordered_map<std::string_view, double>    TokenWeightMap;
    typedef std::pair<std::string_view, double>             TokenWeightPair;

    struct TokenRange { uint32_t Offset, Size; };
    struct TokenWeightLimits { TokenWeightPair Min, Max; };
	
    int32_t term_frequency              (const std::string_view & text, const std::span<const TokenRange> & tokenRanges, TokenWeightMap & frequencies, TokenWeightLimits & limits);
    int32_t term_frequency              (const std::string_view & text, const std::span<const TokenRange> & tokenRanges, std::span<std::string_view> tokenViews, TokenWeightMap & frequencies, TokenWeightLimits & limits);
    int32_t term_frequency              (const std::span<const std::string_view> & tokenViews, TokenWeightMap & frequencies, TokenWeightLimits & limits);
    int32_t tokenize                    (const std::string_view & text, std::vector<TokenRange> & tokenRanges, const std::string_view & terminator = ""); // Returns the position after the last token
    int32_t token_views                 (const std::string_view & document, const std::span<const TokenRange> & tokenRanges, std::span<std::string_view> views);
    int32_t inverse_document_frequency  (const std::span<std::vector<std::string_view>> & documents, TokenWeightMap & idfScores, TokenWeightMap & occurrences);
    int32_t weight_terms
        ( const TokenWeightMap & term_freq_map
        , const TokenWeightMap & inverse_doc_freq_map
        , TokenWeightMap & weighted_terms
        );
    double  cosine_similarity
        ( const TokenWeightMap & tf_idf_1
        , const TokenWeightMap & tf_idf_2
        , double epsilon = 1e-6
        );

    int32_t tokenize                    (const std::span<const std::string_view> & documents, std::span<std::vector<TokenRange>> tokenRanges, const std::string_view & terminator = "");
    int32_t term_frequency              (const std::span<const std::string_view> & documents, const std::span<std::vector<TokenRange>> tokenRanges, std::span<TokenWeightMap> frequencies, std::span<TokenWeightLimits> limits);
    int32_t term_frequency              (const std::span<const std::string_view> & documents, const std::span<std::vector<TokenRange>> tokenRanges, const std::span<std::vector<std::string_view>> tokenViews, std::span<TokenWeightMap> frequencies, std::span<TokenWeightLimits> limits);
    int32_t load_docs                   (const std::span<const std::string_view> & documents, TokenWeightMap & idf_scores, std::span<TokenWeightMap> weighted);
} // namespace 

#endif // LLAI_RANKING_H
