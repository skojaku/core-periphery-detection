import cpnet
import networkx as nx


def test_BE():
    # BE =========================
    G=nx.karate_club_graph()
    print("Running BE algorithm ...")
    be = cpnet.BE()
    be.detect(G)
    pair_id = be.get_pair_id()
    coreness = be.get_coreness()
    sig_pair_id, sig_coreness, significance, p_values = cpnet.qstest(pair_id, coreness, G, be)

def test_MINRES():
    # MINRES =========================
    G=nx.karate_club_graph()
    print("Running MINRES algorithm ...")
    mrs = cpnet.MINRES()
    mrs.detect(G)
    pair_id = mrs.get_pair_id()
    coreness = mrs.get_coreness()
    sig_pair_id, sig_coreness, significance, p_values = cpnet.qstest(pair_id, coreness, G, mrs)
    
def test_LowRankCore():
    # LowRankCore =========================
    G=nx.karate_club_graph()
    print("Running LowRankCore algorithm ...")
    lrc = cpnet.LowRankCore()
    lrc.detect(G)
    pair_id = lrc.get_pair_id()
    coreness = lrc.get_coreness()
    sig_pair_id, sig_coreness, significance, p_values = cpnet.qstest(pair_id, coreness, G, lrc)
    
def test_LapCore():
    # LapCore =========================
    G=nx.karate_club_graph()
    print("Running LapCore algorithm ...")
    lc = cpnet.LapCore()
    lc.detect(G)
    pair_id = lc.get_pair_id()
    coreness = lc.get_coreness()
    sig_pair_id, sig_coreness, significance, p_values = cpnet.qstest(pair_id, coreness, G, lc)
    
def test_LapSgnCore():
    # LapSgnCore =========================
    G=nx.karate_club_graph()
    print("Running LapSgnCore algorithm ...")
    lsc = cpnet.LapSgnCore()
    lsc.detect(G)
    pair_id = lsc.get_pair_id()
    coreness = lsc.get_coreness()
    sig_pair_id, sig_coreness, significance, p_values = cpnet.qstest(pair_id, coreness, G, lsc)
    
def test_Rombach():
    # Rombach =========================
    G=nx.karate_club_graph()
    print("Running Rombach's algorithm ...")
    rb = cpnet.Rombach()
    rb.detect(G)
    pair_id = rb.get_pair_id()
    coreness = rb.get_coreness()
    sig_pair_id, sig_coreness, significance, p_values = cpnet.qstest(pair_id, coreness, G, rb)
    
def test_Rossa():
    # Rossa =========================
    G=nx.karate_club_graph()
    print("Running Rossa's algorithm ...")
    rs = cpnet.Rossa()
    rs.detect(G)
    pair_id = rs.get_pair_id()
    coreness = rs.get_coreness()
    sig_pair_id, sig_coreness, significance, p_values = cpnet.qstest(pair_id, coreness, G, rs)
    
def test_KM_ER():
    # KM--ER =========================
    G=nx.karate_club_graph()
    print("Running KM_ER ...")
    km = cpnet.KM_ER()
    km.detect(G)
    pair_id = km.get_pair_id()
    coreness = km.get_coreness()
    sig_pair_id, sig_coreness, significance, p_values = cpnet.qstest(pair_id, coreness, G, km)
    
def test_KM_config():
    # KM--config =========================
    G=nx.karate_club_graph()
    print("Running KM_config ...")
    km = cpnet.KM_config()
    km.detect(G)
    pair_id = km.get_pair_id()
    coreness = km.get_coreness()
    sig_pair_id, sig_coreness, significance, p_values = cpnet.qstest(pair_id, coreness, G, km)
def test_Surprise():
    # Surprise =========================
    G=nx.karate_club_graph()
    print("Running Surprise ...")
    spr = cpnet.Surprise()
    spr.detect(G)
    pair_id = spr.get_pair_id()
    coreness = spr.get_coreness()
    sig_pair_id, sig_coreness, significance, p_values = cpnet.qstest(pair_id, coreness, G, spr)
	

test_Rossa()
test_BE()
test_MINRES()
test_LowRankCore()
test_LapCore()
test_LapSgnCore()
test_Rombach()
test_KM_ER()
test_KM_config()
test_Surprise()
