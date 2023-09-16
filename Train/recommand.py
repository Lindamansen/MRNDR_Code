import requests
from bs4 import BeautifulSoup

# 定义要搜索的药物名称列表（长度小于20）
drugs = [
    "methoserpidine", "salicylamide", "hydrotalcite", "melatonin", "18,19-dihydroetorphine",
    "tipapkinogene sovacivec", "turmeric extract", "fresolimumab", "n-chlorotaurine", "fluconazole",
    "chloramphenicol", "ecgonine", "4-amino-3-phenylbutyric acid", "zinc sulfate",
    "homatropine methylbromide", "clarithromycin", "insulin detemir", "thiram",
    "2,5-di-tert-butylhydroquinone", "clothiapine", "almasilate", "prochlorperazine",
    "levopropoxyphene", "hydrocortamate", "dexchlorpheniramine", "ximelagatran",
    "polyestradiol phosphate", "benzoic acid", "vinorelbine", "liraglutide",
    "florbetapir f 18", "benfluorex", "gemcitabine", "oxazolidinones", "proquazone",
    "msh, 4-nle-7-phe-alpha-", "bromazepam", "nefazodone", "bacillus subtilis",
    "thiamylal", "ibopamine", "isoflurane", "didecyldimethylammonium", "nelotanserin",
    "mianserin", "tribenoside", "thioridazine", "10-carboxymethyl-9-acridanone",
    "capecitabine", "lasmiditan", "temocillin", "lanreotide", "pateclizumab", "ozoralizumab",
    "carbinoxamine", "doxorubicin", "glipizide", "thymogen", "oxymetholone", "perazine",
    "aminacrine", "triciribine", "asparaginase erwinia chrysanthemi", "muscimol",
    "19-norandrostenedione", "docetaxel anhydrous", "flurbiprofen", "aspartic acid", "azathioprine",
    "pyrithyldione",
]

# 遍历药物列表
for drug in drugs:
    # 构建PubMed搜索的查询字符串
    query = f"{drug} AND advanced breast cancer"

    # 构建PubMed搜索的URL
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={query}"

    # 发送HTTP请求
    response = requests.get(pubmed_url)

    # 使用BeautifulSoup解析页面
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找匹配结果的元素
    result_count = soup.find("span", {"class": "value"})

    # 提取结果数量
    if result_count:
        count = int(result_count.text.replace(",", ""))
        if count > 2:
            print(f"药物 '{drug}' 与 advanced breast cancer 相关的文献数量超过5：{count} 篇")
