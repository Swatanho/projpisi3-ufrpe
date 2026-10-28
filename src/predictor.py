'''Preditor de DCV reutilizavel pelo app HeartHealth (projdsi-ufrpe).  
 
Este modulo NAO treina nada: apenas carrega o artefato produzido por  
src/train_model.py (data/modelo/dcv_modelo.pkl) e prediz pacientes.  
 
Reuso no outro repositorio:  
1. Copiar 'src/predictor.py' e 'data/modelo/dcv_modelo.pkl';  
2. Carregar e prever um paciente:  
 
    from predictor import PredictorDCV  
    pred = PredictorDCV.carregar('dcv_modelo.pkl')  
    resultado = pred.prever(dicionario_do_paciente)  
 
O dicionario do paciente deve conter EXATAMENTE as 20 features do  
contrato (a lista oficial fica disponivel em pred.features).  
 
Classificacao em faixas heuristicas (o app pode rotular como quiser):  
- 'alto'    : probabilidade no threshold de triagem ou acima dele;  
- 'moderado': probabilidade na metade do threshold ou acima dela;  
- 'baixo'   : abaixo disso.  
'''  
 
from pathlib import Path  
 
import joblib  
import numpy as np  
import pandas as pd  
import xgboost as xgb  
 
 
class PacienteInvalido(ValueError):  
    '''Erro de contrato: features ausentes ou desconhecidas no paciente.'''  
 
 
class PredictorDCV:  
    '''Carrega o artefato treinado e prediz o risco de DCV por paciente.'''  
 
    ARTEFATO_PADRAO = (  
        Path(__file__).resolve().parent.parent / 'data' / 'modelo' / 'dcv_modelo.pkl'  
    )  
 
    def __init__(self, artefato):  
        self._modelo = artefato['modelo']  
        self.features = list(artefato['features'])  
        self.threshold = float(artefato['threshold'])  
        self.versao = artefato.get('versao', 'desconhecida')  
        self._winsor_bmi = tuple(artefato['winsor_bmi'])  
        self.metricas_teste = artefato.get('metricas_teste', dict()) 
 
    @classmethod  
    def carregar(cls, caminho=None):  
        '''Carrega o artefato joblib (padrao: data/modelo/dcv_modelo.pkl).'''  
        destino = Path(caminho) if caminho is not None else cls.ARTEFATO_PADRAO  
        if not destino.exists():  
            raise FileNotFoundError(  
                'Artefato nao encontrado em: {} . '.format(destino)  
                + 'Execute primeiro src/train_model.py.'  
            )  
        return cls(joblib.load(destino))  
 
    def _validar(self, dados):  
        '''Exige exatamente as features do contrato (nem mais, nem menos).'''  
        faltando = [f for f in self.features if f not in dados.columns]  
        desconhecidas = [c for c in dados.columns if c not in self.features]  
        if faltando or desconhecidas:  
            partes = []  
            if faltando:  
                partes.append('ausentes: ' + ', '.join(faltando))  
            if desconhecidas:  
                partes.append('desconhecidas: ' + ', '.join(desconhecidas))  
            raise PacienteInvalido(  
                'Paciente fora do contrato de 20 features (' + '; '.join(partes) + ')'  
            )  
 
    def _matriz(self, dados):  
        X = dados[self.features].astype(float).copy()  
        # Reaplica os limites de winsorizacao aprendidos no treino (so o BMI)  
        X['BMI'] = X['BMI'].clip(self._winsor_bmi[0], self._winsor_bmi[1])  
        return xgb.DMatrix(X, feature_names=self.features)  
 
    def _probabilidades(self, dados):  
        dm = self._matriz(dados)  
        it = getattr(self._modelo, 'best_iteration', None)  
        if it is None:  
            return self._modelo.predict(dm)  
        return self._modelo.predict(dm, iteration_range=(0, it + 1))  
 
    def _classificar(self, prob):  
        if prob >= self.threshold:  
            return 'alto'  
        if prob >= self.threshold / 2:  
            return 'moderado'  
        return 'baixo' 
 
    def prever(self, paciente):  
        '''Prediz um unico paciente (dict com as 20 features do contrato).  
 
        Retorna probabilidade, classificacao e os principais fatores de  
        risco (contribuicoes SHAP do proprio paciente, em escala de logito).  
        '''  
        df = pd.DataFrame([paciente])  
        self._validar(df)  
 
        dm = self._matriz(df)  
        it = getattr(self._modelo, 'best_iteration', None)  
        if it is None:  
            prob = float(self._modelo.predict(dm)[0])  
            contribs = self._modelo.predict(dm, pred_contribs=True)[0]  
        else:  
            faixa = (0, it + 1)  
            prob = float(self._modelo.predict(dm, iteration_range=faixa)[0])  
            contribs = self._modelo.predict(dm, pred_contribs=True, iteration_range=faixa)[0]  
 
        valores = df.iloc[0][self.features]  
        fatores = []  
        for nome, c in zip(self.features, contribs[:-1]):  
            fatores.append(  
                {  
                    'feature': nome,  
                    'valor': float(valores[nome]),  
                    'contribuicao': float(c),  
                    'efeito': 'aumenta' if c > 0 else 'reduz',  
                }  
            )  
        fatores.sort(key=lambda f: abs(f['contribuicao']), reverse=True)  
 
        return {  
            'probabilidade': prob,  
            'classificacao': self._classificar(prob),  
            'threshold': self.threshold,  
            'fatores_principais': fatores[:5],  
            'versao': self.versao,  
        }  
 
    def prever_df(self, dados):  
        '''Predicao em lote (DataFrame com as 20 colunas do contrato).'''  
        self._validar(dados)  
        prob = self._probabilidades(dados)  
        return pd.DataFrame(  
            {  
                'probabilidade': prob,  
                'classificacao': [self._classificar(p) for p in prob],  
            },  
            index=dados.index,  
        ) 
