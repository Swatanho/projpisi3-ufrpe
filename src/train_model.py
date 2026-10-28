'''Treinamento supervisionado do modelo de predicao de DCV (HeartHealth).  
 
Pergunta do projeto: com base nas informacoes coletadas previamente, o  
paciente tem (ou desenvolvera) doenca cardiaca? Como o BRFSS 2015 e um  
inquerito transversal e autorrelatado, o modelo estima a probabilidade de  
o perfil corresponder a alguem COM doenca cardiovascular (triagem), e nao  
a incidencia futura.  
 
Decisoes de projeto:  
- Features: as 20 variaveis do contrato do app (DSI), a mesma matriz do  
  clustering; exclui o alvo, Diabetes_binary, colunas derivadas e is_outlier.  
- Desbalanceamento (cerca de 10,3 positivos em cada 100): scale_pos_weight.  
- Threshold operacional de triagem: maximo recall sujeito a precisao  
  >= 2x a prevalencia; fallback para F2 se nenhum threshold qualificar.  
- Split estratificado 80/20 (teste tocado uma unica vez); validacao interna  
  de 10 por cento do treino para early stopping e escolha do threshold.  
- Baseline interpretavel: regressao logistica com class_weight balanced.  
 
Saidas em data/modelo/ (fora do versionamento):  
- dcv_modelo.pkl       : artefato joblib consumido pelo app (src/predictor.py)  
- metricas_modelo.json : relatorio completo (CV, baseline, teste)  
'''  
 
from pathlib import Path  
import json  
 
import joblib  
import numpy as np  
import pandas as pd  
import xgboost as xgb  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import (  
    average_precision_score,  
    brier_score_loss,  
    confusion_matrix,  
    f1_score,  
    fbeta_score,  
    precision_score,  
    recall_score,  
    roc_auc_score,  
)  
from sklearn.model_selection import StratifiedKFold, train_test_split  
 
BASE_DIR = Path(__file__).resolve().parent.parent  
DATA_PATH = BASE_DIR / 'data' / 'tratados' / 'diabetes_processed.parquet'  
OUTPUT_DIR = BASE_DIR / 'data' / 'modelo'  
 
TARGET_COL = 'HeartDiseaseorAttack'  
 
# Contrato de 20 features do app (DSI) - identico ao usado no clustering  
FEATURES = [  
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',  
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump',  
    'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 'MentHlth', 'PhysHlth',  
    'DiffWalk', 'Sex', 'Age', 'Education', 'Income',  
]  
 
SEED = 42  
VERSION = '1.0.0'  
TEST_SIZE = 0.20        # hold-out final, tocado uma unica vez  
VAL_FRAC = 0.10         # fracao do treino p/ early stopping e threshold  
CV_FOLDS = 5  
PISO_PREVALENCIA = 2.0  # regra de triagem: precisao minima = 2x prevalencia 
 
 
def carregar_base():  
    if not DATA_PATH.exists():  
        raise FileNotFoundError(  
            'Base tratada nao encontrada em: {} . Execute src/data_prep.py.'.format(DATA_PATH)  
        )  
    df = pd.read_parquet(DATA_PATH)  
    faltando = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]  
    if faltando:  
        raise ValueError('Colunas ausentes na base tratada: {}'.format(faltando))  
    return df  
 
 
def metricas_em(y_true, prob, threshold):  
    '''Metricas em um dado threshold, alem das curvas globais (AUCs, Brier).'''  
    y = np.asarray(y_true).astype(int)  
    prob = np.asarray(prob, dtype=float)  
    pred = prob >= threshold  
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()  
    return {  
        'threshold': float(threshold),  
        'prevalencia': float(y.mean()),  
        'recall': float(recall_score(y, pred, zero_division=0)),  
        'precisao': float(precision_score(y, pred, zero_division=0)),  
        'f1': float(f1_score(y, pred, zero_division=0)),  
        'f2': float(fbeta_score(y, pred, beta=2, zero_division=0)),  
        'acuracia': float((pred == y).mean()),  
        'roc_auc': float(roc_auc_score(y, prob)),  
        'pr_auc': float(average_precision_score(y, prob)),  
        'brier': float(brier_score_loss(y, prob)),  
        'prob_media_prevista': float(prob.mean()),  
        'confusao': {'vn': int(tn), 'fp': int(fp), 'fn': int(fn), 'vp': int(tp)},  
    }  
 
 
def treinar_xgb(X_tr, y_tr, X_val, y_val):  
    '''XGBoost com peso de classe e early stopping na validacao (AUCPR).'''  
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))  
    params = {  
        'objective': 'binary:logistic',  
        'eval_metric': 'aucpr',  
        'max_depth': 6,  
        'learning_rate': 0.1,  
        'subsample': 0.8,  
        'colsample_bytree': 0.8,  
        'min_child_weight': 5,  
        'scale_pos_weight': spw,  
        'seed': SEED,  
        'verbosity': 0,  
    }  
    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURES)  
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURES)  
    return xgb.train(  
        params,  
        dtr,  
        num_boost_round=1000,  
        evals=[(dval, 'val')],  
        early_stopping_rounds=50,  
        verbose_eval=False,  
    )  
 
 
def prever(modelo, X):  
    '''Probabilidades no melhor ponto do early stopping.'''  
    dm = xgb.DMatrix(X, feature_names=FEATURES)  
    it = getattr(modelo, 'best_iteration', None)  
    if it is None:  
        return modelo.predict(dm)  
    return modelo.predict(dm, iteration_range=(0, it + 1)) 
 
 
def escolher_threshold(y_true, prob, piso_precisao):  
    '''Regra de triagem: maior recall com precisao acima do piso.  
 
    Varre thresholds de 0.01 a 0.99; entre os que respeitam o piso de  
    precisao fica com o de maior recall (empate: o maior threshold).  
    Se nenhum qualificar, cai para o F2 maximo.  
    '''  
    y = np.asarray(y_true).astype(int)  
    grid = np.round(np.linspace(0.01, 0.99, 197), 3)  
    melhor_t = None  
    melhor_recall = -1.0  
    for t in grid:  
        pred = prob >= t  
        if precision_score(y, pred, zero_division=0) >= piso_precisao:  
            r = recall_score(y, pred, zero_division=0)  
            if r > melhor_recall + 1e-12:  
                melhor_t = float(t)  
                melhor_recall = r  
    if melhor_t is not None:  
        return melhor_t, 'recall maximo com precisao acima de 2x a prevalencia'  
    melhor_t = 0.5  
    melhor_f2 = -1.0  
    for t in grid:  
        f2 = fbeta_score(y, prob >= t, beta=2, zero_division=0)  
        if f2 > melhor_f2:  
            melhor_t = float(t)  
            melhor_f2 = f2  
    return melhor_t, 'fallback: F2 maximo (piso de precisao inatingivel)'  
 
 
def _resumo_cv(cv):  
    '''Media e desvio das metricas dos folds.'''  
    resumo = {}  
    for chave in ['recall', 'precisao', 'f2', 'pr_auc', 'roc_auc']:  
        valores = np.array([m[chave] for m in cv])  
        resumo[chave] = {'media': float(valores.mean()), 'desvio': float(valores.std())}  
    return resumo  
 
 
def executar():  
    df = carregar_base()  
    X = df[FEATURES].astype(float)  
    y = df[TARGET_COL].astype(int)  
 
    print('=' * 60)  
    print(' TREINAMENTO DO MODELO DE PREDICAO DE DCV (XGBoost)')  
    print('=' * 60)  
    print('Registros: {} | Prevalencia do alvo: {:.4f}'.format(len(df), y.mean()))  
    print('Features do contrato: {}'.format(len(FEATURES)))  
 
    X_tr, X_te, y_tr, y_te = train_test_split(  
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED  
    )  
    X_in, X_val, y_in, y_val = train_test_split(  
        X_tr, y_tr, test_size=VAL_FRAC, stratify=y_tr, random_state=SEED  
    )  
    print('Treino interno: {} | Validacao: {} | Teste: {}'.format(len(X_in), len(X_val), len(X_te)))  
 
    # Limites de winsorizacao do BMI (persistidos p/ reuso na inferencia)  
    winsor_bmi = [float(df['BMI'].quantile(0.001)), float(df['BMI'].quantile(0.995))] 
 
 
    print('[1/4] Baseline: regressao logistica (class_weight balanced)')  
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=SEED)  
    lr.fit(X_in, y_in)  
    prob_lr_val = lr.predict_proba(X_val)[:, 1]  
 
    print('[2/4] XGBoost (scale_pos_weight + early stopping na validacao)')  
    modelo = treinar_xgb(X_in, y_in, X_val, y_val)  
    print('melhor iteracao: {}'.format(modelo.best_iteration))  
    prob_val = prever(modelo, X_val)  
 
    piso = PISO_PREVALENCIA * float(y_val.mean())  
    threshold, regra = escolher_threshold(y_val, prob_val, piso)  
    print('[3/4] Threshold de triagem: {:.2f} ({})'.format(threshold, regra))  
    print('   piso de precisao adotado: {:.3f}'.format(piso))  
 
    print('[4/4] Validacao cruzada de {} folds (no threshold de triagem)'.format(CV_FOLDS))  
    cv = []  
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)  
    for fold, (idx_tr, idx_va) in enumerate(skf.split(X_tr, y_tr), start=1):  
        m = treinar_xgb(X_tr.iloc[idx_tr], y_tr.iloc[idx_tr], X_tr.iloc[idx_va], y_tr.iloc[idx_va])  
        prob_fold = prever(m, X_tr.iloc[idx_va])  
        met = metricas_em(y_tr.iloc[idx_va], prob_fold, threshold)  
        cv.append(met)  
        r_fold = met['recall']  
        pr_fold = met['pr_auc']  
        print('   fold {}: recall {:.3f} / pr_auc {:.3f}'.format(fold, r_fold, pr_fold))  
 
    prob_te = prever(modelo, X_te)  
    met_teste = metricas_em(y_te, prob_te, threshold)  
    met_teste_050 = metricas_em(y_te, prob_te, 0.5)  
    met_lr_val = metricas_em(y_val, prob_lr_val, threshold)  
    met_lr_teste = metricas_em(y_te, lr.predict_proba(X_te)[:, 1], threshold)  
    met_val = metricas_em(y_val, prob_val, threshold)  
 
    print('')  
    print('=' * 60)  
    print(' RESULTADO NO TESTE (hold-out de 20 por cento, nunca visto)')  
    print('=' * 60)  
    print('Threshold de triagem : {:.2f}'.format(threshold))  
    print('Recall / Precisao    : {:.3f} / {:.3f}'.format(met_teste['recall'], met_teste['precisao']))  
    print('F1 / F2              : {:.3f} / {:.3f}'.format(met_teste['f1'], met_teste['f2']))  
    print('ROC-AUC / PR-AUC     : {:.3f} / {:.3f}'.format(met_teste['roc_auc'], met_teste['pr_auc']))  
    print('Brier (calibracao)   : {:.3f}'.format(met_teste['brier']))  
    c = met_teste['confusao']  
    print('Matriz de confusao   : VN={} FP={} FN={} VP={}'.format(c['vn'], c['fp'], c['fn'], c['vp']))  
 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  
    artefato = {  
        'modelo': modelo,  
        'features': FEATURES,  
        'threshold': float(threshold),  
        'winsor_bmi': winsor_bmi,  
        'prevalencia_base': float(y.mean()),  
        'versao': VERSION,  
        'metricas_teste': met_teste,  
    }  
    caminho_pkl = OUTPUT_DIR / 'dcv_modelo.pkl'  
    joblib.dump(artefato, caminho_pkl)  
 
    relatorio = {  
        'versao': VERSION,  
        'seed': SEED,  
        'n_registros': int(len(df)),  
        'n_features': len(FEATURES),  
        'features': FEATURES,  
        'prevalencia_base': float(y.mean()),  
        'winsor_bmi': winsor_bmi,  
        'threshold': float(threshold),  
        'regra_threshold': regra,  
        'piso_precisao': float(piso),  
        'melhor_iteracao': int(modelo.best_iteration),  
        'cv_folds': cv,  
        'cv_resumo': _resumo_cv(cv),  
        'xgb_validacao': met_val,  
        'xgb_teste': met_teste,  
        'xgb_teste_threshold_050': met_teste_050,  
        'baseline_logreg_validacao': met_lr_val,  
        'baseline_logreg_teste': met_lr_teste,  
    }  
    caminho_json = OUTPUT_DIR / 'metricas_modelo.json'  
    caminho_json.write_text(  
        json.dumps(relatorio, indent=2, ensure_ascii=False, default=float),  
        encoding='utf-8',  
    )  
 
    print('')  
    print('Artefato salvo em  : {}'.format(caminho_pkl))  
    print('Relatorio salvo em : {}'.format(caminho_json))  
    return relatorio  
 
 
if __name__ == '__main__':  
    executar() 
