# Preliminary semantic interpretation of the current clusters

These labels were produced from the automatically generated cluster dossiers using the
offline TF-IDF/SVD fallback. They are useful for inspection, but should be regenerated
with multilingual sentence embeddings before being treated as final manuscript results.

| Cluster | Hosts | Preliminary label | Evidence / interpretation | Main outgoing pattern |
|---|---:|---|---|---|
| C0 | 19 | French alternative/local media | Representative pages are French local/alternative-news and activist information sites; top terms include French magazine/news vocabulary. | Very few/no observed outgoing cross-host links in this sample. |
| C1 | 82 | Mixed community/security/general services | Representatives include a security-engineering blog, DEF CON, Leftychan, and miscellaneous services. This cluster is heterogeneous and likely reflects limitations of the fallback embedding. | Mainly links toward C5 and C8. |
| C2 | 14 | Russian-language darknet portals/marketplaces | Russian text dominates; representative pages include BAZA/OMG-related material and Russian onion link portals. | Mainly toward C8 and C5, with some internal links. |
| C3 | 13 | Drug storefronts | Representative pages explicitly advertise cannabis, cocaine, psychedelics and related products. | Few/no observed outgoing cross-host links. |
| C4 | 22 | OnionMail / encrypted mail services | Highly homogeneous OnionMail templates with sign-up/logon/help pages. | Few/no observed outgoing cross-host links. |
| C5 | 76 | Mixed directories/tools/anonymous services | Includes Tor directory entries, anonymous image hosting, ransomware-site directories and other utility pages. Heterogeneous. | Mostly internal and toward C8. |
| C6 | 15 | German-language news/hackspace/politics | German text dominates; representatives include Krautspace, taz and Datenkanal. | Sparse links, mainly toward C5/C8. |
| C7 | 8 | Tor navigation directories | Dominated by tor.taxi-like directory/navigation pages. | Strongly points toward C8, C5 and C1. |
| C8 | 69 | Mixed privacy/technical/commercial services | Representatives include VPN, whistleblower software and advertised hacking services. Heterogeneous; multilingual embeddings should likely split it further. | Sparse internal links and links to C10/C9. |
| C9 | 11 | JavaScript-dependent / interactive services | Representatives mainly contain “enable JavaScript/loading” content such as mail and exchange services. This may be a presentation/technology cluster rather than a topic cluster. | Few/no observed outgoing links. |
| C10 | 47 | Tor search engines and link directories | Representatives are Hidden Wiki, Amnesia search, tormarks and other large link lists. | Major routing hub: very strong outgoing links to C5, C8 and C1, and also to most other clusters. |
| C11 | 7 | Carding / stolen-payment-data storefronts | Representatives explicitly advertise CVV/fullz/dumps/checking services. | Links primarily to C10 and C5, with some internal links. |

## Important interpretation

The strongest domain-level pattern is the distinction between *content providers/services*
and *navigation/directory hubs*. In particular, C10 and C7 act as gateways pointing toward
many other semantic groups, whereas clusters such as drug storefronts (C3), OnionMail (C4),
and several media/content clusters have relatively little observed outgoing cross-host traffic.

The mixed clusters C1, C5, C8 and the technology-driven C9 should not be given strong semantic
labels from the fallback TF-IDF/SVD representation. The recommended final run uses a multilingual
sentence-transformer so German, Russian, French and English pages are embedded by meaning rather
than mostly by shared vocabulary.
