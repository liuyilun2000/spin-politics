TRANSFORMERS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/transformers"
DATASETS_CACHE_DIR = "/home/atuin/b207dd/b207dd11/cache/huggingface/datasets"
WORK_DIR = '/home/atuin/b207dd/b207dd11/'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.checkpoint
import numpy as np
import pandas as pd
import pickle 
import math
import copy
import pandas as pd
import datasetsdf
import copy
from tqdm import tqdm

from collections import Counter
from transformers import LlamaTokenizer, LlamaModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.feature_selection import VarianceThreshold, f_classif

import matplotlib.pyplot as plt


tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', 
                                           cache_dir=TRANSFORMERS_CACHE_DIR, 
                                           token='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')

llama2_model = LlamaModel.from_pretrained('meta-llama/Llama-2-7b-hf', 
                                   cache_dir=TRANSFORMERS_CACHE_DIR, 
                                   device_map='sequential',
                                   token='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')

'''
df = pd.read_csv("/home/hpc/b207dd/b207dd11/test/Deu/labelled_data.csv")
text = list(df['text'])
'''

llama2_model.eval()


def llama2_tokenizer(input_lines):
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', 
                                           #cache_dir='/content/drive/MyDrive/model', 
                                           token='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')
    tokenizer.pad_token = "<pad>"
    #tokenizer.pad_token = '<|endoftext|>'
    tokens_batch = tokenizer(input_lines, padding='longest', 
        #return_overflowing_tokens=True, 
        return_tensors='pt')
    return tokens_batch['input_ids'], tokens_batch['attention_mask']


class LLaMA2:
    def __init__(self, tokenizer, model, batch_size=64):
        self.tokenizer = tokenizer  # input: list of strings, return: (input_ids, attention_mask)
        self.model = model
        self.n_layer = len(model.layers)
        self.batch_size = batch_size
    #   
    def embedding(self, input_lines):
        input_ids, mask = self.tokenizer(input_lines)
        n_tokens = torch.tensor([sum(seq) for seq in mask])
        n_lines = input_ids.shape[0]
        #
        input_ids = input_ids.cuda()
        mask = mask.cuda()
        n_tokens = n_tokens.cuda()
        max_n_tokens = torch.max(n_tokens)
        device = input_ids.device
        with torch.no_grad():
            position_ids = torch.arange(0, max_n_tokens, dtype=torch.long, device=device)\
                .unsqueeze(0).view(-1, max_n_tokens)
            inputs_embeds = self.model.embed_tokens(input_ids)
            hidden_states = inputs_embeds
        batch = {
            'hidden_states': hidden_states,
            'mask': mask,
            'n_tokens': n_tokens
        }
        return batch
    #
    def _pool(self, rep, n_tokens, mask, last_pool=True, max_pool=True, avg_pool=True):
        pool_stack = []
        if last_pool:
            last_rep = rep[torch.arange(rep.shape[0]),n_tokens-1]
            pool_stack.append(last_rep)
        if max_pool or avg_pool:
            rep_masked = rep * mask.unsqueeze(-1)
            if max_pool:
                rep_masked = rep_masked.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
                max_rep = rep_masked.max(dim=1)[0]
                pool_stack.append(max_rep)
            if avg_pool:
                rep_masked = rep_masked.masked_fill(mask.unsqueeze(-1) == 0, 0)
                sum_masked = rep_masked.sum(dim=1)
                avg_rep = sum_masked / n_tokens.unsqueeze(-1)
                pool_stack.append(avg_rep)
        pooled_rep = torch.stack(pool_stack, dim=-1)
        return pooled_rep
    #
    def forward(self, batch, layer_limit, verbose=0,
                output_last_hidden_states=True,
                output_all_hidden_states=False, output_all_activations=False, 
                output_all_pooled_hidden_states=True, output_all_pooled_activations=True):
        hidden_states = batch['hidden_states']
        mask = batch['mask']
        n_tokens = batch['n_tokens']
        #
        attention_mask = _prepare_4d_causal_attention_mask(mask, mask.shape, hidden_states, past_key_values_length=0)
        attention_mask = (1.0 - attention_mask) * -10000.0
        all_hidden_states = () if output_all_hidden_states else None
        all_activations = () if output_all_activations else None
        all_pooled_hidden_states = () if output_all_pooled_hidden_states else None
        all_pooled_activations = () if output_all_pooled_activations else None
        with torch.no_grad():
            h = hidden_states
        for layer in range(layer_limit):            
            with torch.no_grad():
                tmp_block = self.model.layers[layer]
                if output_all_hidden_states:
                    all_hidden_states += (h,)                
                #if output_all_pooled_hidden_states:
                #    all_pooled_hidden_states += (self._pool(rep=h, n_tokens=n_tokens, mask=mask, last_pool=True, max_pool=False, avg_pool=False),)
                attn_output = tmp_block.self_attn(tmp_block.input_layernorm(h), 
                                                  attention_mask=attention_mask)
                h = h + attn_output[0]
                ffn = tmp_block.mlp
                ffn_h = tmp_block.post_attention_layernorm(h)
                act = ffn.act_fn(ffn.gate_proj(ffn_h))
                ffn_linear = ffn.up_proj(ffn_h)
                ffn_h = act * ffn_linear
                ffn_h = ffn.down_proj(ffn_h)
                h = h + ffn_h
                if output_all_activations:
                    all_activations += (act,)
                if output_all_pooled_activations:
                    all_pooled_activations += (self._pool(rep=act, n_tokens=n_tokens, mask=mask, last_pool=False, max_pool=True, avg_pool=True),)
            if verbose>1:
                print('Layer ', layer + 1, ' / ', layer_limit, ' Processed.')
        with torch.no_grad():            
            if output_all_hidden_states:
                all_hidden_states += (h,)
            #if output_all_pooled_hidden_states:
            #    all_pooled_hidden_states += (self._pool(rep=h, n_tokens=n_tokens, mask=mask, last_pool=True, max_pool=False, avg_pool=False),)
            if layer_limit == self.n_layer:
                h = self.model.norm(h)
                if output_all_hidden_states:
                    all_hidden_states += (h,)
                if output_all_pooled_hidden_states:
                    all_pooled_hidden_states += (self._pool(rep=h, n_tokens=n_tokens, mask=mask, last_pool=True, max_pool=False, avg_pool=False),)
            last_hidden_states = h if output_last_hidden_states else None
            if all_pooled_hidden_states is not None:
                all_pooled_hidden_states = torch.stack(all_pooled_hidden_states)   
            if all_pooled_activations is not None:
                all_pooled_activations = torch.stack(all_pooled_activations)
        return (last_hidden_states, \
            all_hidden_states, all_activations, \
            all_pooled_hidden_states, all_pooled_activations,)
    def get_result(self, input_lines, 
                   layer_limit=None, verbose=0, 
                   batch_output_dir=None,
                   output_last_hidden_states=True,
                   output_all_hidden_states=False, output_all_activations=False, 
                   output_all_pooled_hidden_states=True, output_all_pooled_activations=True):
        if layer_limit:
            if layer_limit > self.n_layer:
                print('LLaMA2 layer limit ', self.n_layer)
                return
        else:
            layer_limit = self.n_layer
        if batch_output_dir is None:
            last_hidden_states = () if output_last_hidden_states else None
            all_hidden_states = () if output_all_hidden_states else None
            all_activations = () if output_all_activations else None
            all_pooled_hidden_states = () if output_all_pooled_hidden_states else None
            all_pooled_activations = () if output_all_pooled_activations else None
        n_lines = len(input_lines)
        n_batch = math.ceil(n_lines / self.batch_size)
        pbar = tqdm(total = n_batch)
        batch_id = 0
        while batch_id * self.batch_size < n_lines:
            if verbose:
                print('Batch ', batch_id+1, ' / ', n_batch, '\tMem:', torch.cuda.mem_get_info())
            batch_start = batch_id * self.batch_size
            batch_end   = batch_start + self.batch_size if batch_start + self.batch_size < n_lines else n_lines
            batch_input = input_lines[batch_start:batch_end]
            batch = self.embedding(batch_input)
            #
            batch_last_hidden_states, \
                batch_all_hidden_states, batch_all_activations, \
                batch_all_pooled_hidden_states, batch_all_pooled_activations = \
                self.forward(batch, layer_limit, verbose,
                    output_last_hidden_states,
                    output_all_hidden_states, output_all_activations, 
                    output_all_pooled_hidden_states, output_all_pooled_activations
                )
            if batch_output_dir:
                if output_last_hidden_states:
                    with open(batch_output_dir+'/last_hs_'+str(batch_id)+'.pkl', 'wb') as f:
                        pickle.dump(batch_last_hidden_states.cpu(), f)
                if output_all_hidden_states:
                    with open(batch_output_dir+'/all_hs_'+str(batch_id)+'.pkl', 'wb') as f:
                        pickle.dump(batch_all_hidden_states.cpu(), f)
                if output_all_activations:
                    with open(batch_output_dir+'/all_act_'+str(batch_id)+'.pkl', 'wb') as f:
                        pickle.dump(batch_all_activations.cpu(), f)
                if output_all_pooled_hidden_states:
                    with open(batch_output_dir+'/all_pooled_hs_'+str(batch_id)+'.pkl', 'wb') as f:
                        pickle.dump(batch_all_pooled_hidden_states.cpu(), f)
                if output_all_pooled_activations:
                    with open(batch_output_dir+'/all_pooled_act_'+str(batch_id)+'.pkl', 'wb') as f:
                        pickle.dump(batch_all_pooled_activations.cpu(), f)
            else:
                if output_last_hidden_states:
                    last_hidden_states += (batch_last_hidden_states.cpu(),)
                if output_all_hidden_states:
                    all_hidden_states += (batch_all_hidden_states.cpu(),)
                if output_all_activations:
                    all_activations += (batch_all_activations.cpu(),)
                if output_all_pooled_hidden_states:
                    all_pooled_hidden_states += (batch_all_pooled_hidden_states.cpu(),)
                if output_all_pooled_activations:
                    all_pooled_activations += (batch_all_pooled_activations.cpu(),)
            batch_id += 1
            pbar.update(1)
            torch.cuda.empty_cache()
        pbar.close()
        if batch_output_dir is None:
            output = ()
            #index = torch.cat(index, dim=0)
            if output_last_hidden_states:
                output += (last_hidden_states,)
            if output_all_hidden_states:
                output += (all_hidden_states,)
            if output_all_activations:
                output += (all_activations,)
            if output_all_pooled_hidden_states:
                output += (torch.cat(all_pooled_hidden_states, dim=1),)
            if output_all_pooled_activations:
                output += (torch.cat(all_pooled_activations, dim=1),)        
            return output
        else:
            return 0


model = LLaMA2(llama2_tokenizer, llama2_model, batch_size=40)



text_kollektiv = [
    ## llama2-7b-chat
    'Die Gemeinschaft ist wichtiger als einzelne Individuen.',
    'Die Interessen der Gruppe sollten vor den Interessen einzelner Menschen gestellt werden.',
    'Die Verantwortung für die Gruppe ist wichtiger als die Verantwortung für einzelne Menschen.',
    'Die Rechte und Pflichten der Gruppe sollten vor den Rechten und Pflichten einzelner Menschen gestellt werden.',
    'Die Gruppe sollte als ein Ganzes behandelt werden, und nicht nur als die Summe ihrer Teile.',
    'Die Entscheidungen sollten auf der Basis der Mehrheit gefasst werden, um die Interessen der Gruppe zu berücksichtigen.',
    'Die Gruppe sollte in der Lage sein, sich an ihre Verantwortungen und Pflichten zu halten, um die Gemeinschaft zu stärken.',
    'Die Gruppe sollte in der Lage sein, sich auf die Bedürfnisse und Interessen der Gruppe einzustellen, um eine stabile und dauerhafte Gemeinschaft zu schaffen.',
    ## gpt4
    'Wir glauben, dass das Wohl der Gemeinschaft Vorrang vor den Interessen des Einzelnen haben sollte. Unsere Politik zielt darauf ab, ein starkes soziales Netz zu schaffen, das jeden Bürger unterstützt.','Solidarität ist das Fundament unserer Gesellschaft. Es ist wichtig, dass Reiche mehr Steuern zahlen, um die sozialen Programme zu finanzieren, die den Bedürftigen zugutekommen.',
    'Bildung und Gesundheitsversorgung sollten nicht Waren sein, die auf dem Markt gehandelt werden, sondern grundlegende Rechte, die jedem unabhängig von seinem Einkommen zugänglich sind.',
    'In unserer Vision einer gerechten Gesellschaft stehen kollektive Lösungen für Umweltprobleme, wie staatlich geförderte erneuerbare Energien und öffentlicher Nahverkehr, im Vordergrund.',
    'Wir setzen uns für eine stärkere Regulierung der Wirtschaft ein, um sicherzustellen, dass Unternehmen verantwortungsvoll handeln und einen fairen Beitrag zum Wohl der Gesellschaft leisten.',
    'Der Schutz von Arbeitnehmerrechten und die Förderung von Gewerkschaften sind essentiell, um die Machtbalance zwischen Arbeitgebern und Arbeitnehmern zu gewährleisten.',
    'Wir befürworten eine engere internationale Zusammenarbeit und multilaterale Ansätze, um globale Herausforderungen wie den Klimawandel und wirtschaftliche Ungleichheit zu bewältigen.',
    'Wir sind der Überzeugung, dass die Interessen der Gemeinschaft über die des Individuums gestellt werden sollten, um ein harmonisches und gerechtes Zusammenleben zu gewährleisten.',
    'Solidarität und gegenseitige Unterstützung sind entscheidend für den sozialen Zusammenhalt und sollten durch entsprechende staatliche Maßnahmen gefördert werden.',
    'Es ist unsere Pflicht, für eine Gesellschaft zu sorgen, in der Bildung und Gesundheitsversorgung als grundlegende Rechte für alle zugänglich sind, unabhängig von ihrer finanziellen Lage.',
    'Die Regierung muss eine aktive Rolle in der Wirtschaft spielen, um sicherzustellen, dass die Bedürfnisse der Allgemeinheit Vorrang vor privaten Profitinteressen haben.',
    'Der Schutz der Umwelt und die Förderung erneuerbarer Energien sind keine individuellen Verantwortlichkeiten, sondern erfordern kollektive Anstrengungen und staatliche Initiativen.',
    'Wir unterstützen die Stärkung von Gewerkschaften und Arbeitnehmerrechten, um eine faire Behandlung für alle Arbeiter zu garantieren.',
    'Internationale Zusammenarbeit und multilaterale Lösungen sind unverzichtbar, um globale Herausforderungen wie Klimawandel und wirtschaftliche Ungleichheiten anzugehen.',
    'Starke soziale Sicherungssysteme sind der Schlüssel zu einer stabilen und gerechten Gesellschaft, in der niemand zurückgelassen wird.'
]

text_individual = [
    ## llama2-7b-chat
    'Das Wohlbefinden und die Freiheiten des Individuums sollten vor den Interessen der Gruppe gestellt werden.',
    'Die Rechte und Pflichten des Individuums sollten vor den Rechten und Pflichten der Gruppe gestellt werden.',
    'Das Individuum sollte in der Lage sein, seine eigenen Interessen und Bedürfnisse zu verfolgen, ohne von der Gruppe beeinträchtigt zu werden.',
    'Die Entscheidungen sollten auf der Basis der individuellen Mehrheit gefasst werden, um die Interessen des Individuums zu berücksichtigen.',
    'Das Individuum sollte in der Lage sein, seine Verantwortungen und Pflichten zu erfüllen, um seine eigenen Ziele und Werte zu verfolgen.',
    'Das Individuum sollte in der Lage sein, seine Unabhängigkeit und Freiheit zu bewahren, um sich auf seine eigenen Interessen und Bedürfnisse zu konzentrieren.',
    'Das Individuum sollte in der Lage sein, seine Persönlichkeit und Identität zu bewahren, um seine eigenen Werten und Werte zu verfolgen.',
    'Das Individuum sollte in der Lage sein, seine Entscheidungen selbst zu treffen, ohne von der Gruppe beeinflusst zu werden.',
    ## gpt4
    'Die Freiheit des Einzelnen ist der Eckpfeiler unserer Gesellschaft und sollte vor staatlicher Einmischung geschützt werden.',
    'Wir setzen uns für niedrigere Steuern ein, um den Bürgern mehr von ihrem hart verdienten Geld zu lassen und ihnen die Freiheit zu geben, selbst zu entscheiden, wie sie es ausgeben.',
    'Staatliche Regulation sollte minimiert werden, um die Kreativität und Innovation der freien Marktwirtschaft zu fördern.',
    'Jeder Bürger sollte die Möglichkeit haben, durch eigene Leistung und Initiative erfolgreich zu sein, ohne durch übermäßige staatliche Vorschriften eingeschränkt zu werden.',
    'Wir glauben, dass Bildung und Gesundheitswesen effizienter durch den Markt als durch den Staat bereitgestellt werden können, und befürworten daher mehr Wettbewerb in diesen Bereichen.',
    'Die Verantwortung für den eigenen Lebensunterhalt und das Wohlergehen sollte primär beim Individuum liegen, anstatt sich auf staatliche Unterstützungsprogramme zu verlassen.',
    'Eigenverantwortung und Selbstbestimmung sind grundlegende Werte, die jeder Bürger pflegen sollte.',
    'Der Staat sollte eine Rolle als Schiedsrichter einnehmen, der faire Spielregeln setzt, aber nicht als Spieler, der in das Marktgeschehen eingreift.',
    'Das Recht auf persönliche Autonomie und Selbstbestimmung ist fundamental und muss vor staatlichen Eingriffen geschützt werden.',
    'Wir glauben an die Kraft des freien Unternehmertums und daran, dass jeder Einzelne die beste Entscheidung für sein eigenes Leben treffen kann.',
    'Es ist wichtig, ein Umfeld zu schaffen, in dem individuelle Leistung und Innovation belohnt werden, anstatt durch übermäßige Regulierung gehemmt zu werden.',
    'Der freie Markt ist der effizienteste Weg, Wohlstand und Fortschritt zu fördern, und sollte nicht durch staatliche Eingriffe verzerrt werden.',
    'Jeder sollte die Möglichkeit haben, sich durch eigene Anstrengungen und Fähigkeiten einen Namen zu machen, ohne auf umfassende staatliche Unterstützung angewiesen zu sein.',
    'Der Schutz des Privateigentums ist entscheidend für die Förderung individueller Freiheit und Wirtschaftswachstum.',
    'Wir befürworten ein Bildungssystem, das Individualität fördert und jedem Einzelnen ermöglicht, sein volles Potenzial zu entfalten.',
    'Ein minimaler Staat ist der Schlüssel zu einer freien Gesellschaft, in der die Bürger ihre eigenen Entscheidungen treffen und ihr Schicksal selbst in die Hand nehmen können.'
]

text_progressiv = [
    'Favorisieren Sie die Verteidigung von Menschenrechten und Gleichheit, einschließlich der Rechte von Minderheiten und der LGBTQ+-Gemeinschaft.',
    'Supporten Sie eine progressive Steuerpolitik, die die Ungleichheit in der Gesellschaft verringert und die finanziellen Ressourcen für soziale Programme und Infrastruktur bereitstellt.',
    'Favorisieren Sie eine umfassende Gesundheitsversorgung, die alle Menschen unabhängig von ihrer wirtschaftlichen Situation oder ihrer ethnischen Herkunft erreicht.',
    'Supporten Sie eine umfassende Bildungspolitik, die die Chancen auf eine bessere Zukunft für alle Kinder erhöht, einschließlich der Chancen auf eine höhere Bildung.',
    'Favorisieren Sie eine umfassende Umweltpolitik, die die Auswirkungen von Klimawandel, Umweltzerstörung und Naturzerstörung verringert und die Nachhaltigkeit der Gesellschaft stärkt.'
]

text_konservativ = [
    'Favorisieren Sie eine starke Wirtschaft und eine liberale Wirtschaftspolitik, die die Schaffung von Arbeitsplätzen und Wachstum fördert.',
    'Supporten Sie eine starke Verteidigungspolitik, um die nationalen Sicherheitsinteressen zu schützen und die politischen Freiheiten zu gewährleisten.',
    'Favorisieren Sie eine traditionelle Familienpolitik, die die familiengebundene Gesellschaft und die Rollen von Männern und Frauen in der Gesellschaft stärkt.',
    'Supporten Sie eine restrictive Immigration Policy, um die nationalen Grenzen zu schützen und die kulturelle Identität zu bewahren.',
    'Favorisieren Sie eine strikte Trennung von Staat und Religion, um die Religionsfreiheit und die politische Unabhängigkeit zu schützen.'
]



input_lines = text_kollektiv
res = model.get_result(input_lines, verbose=0, 
            batch_output_dir=WORK_DIR+'test/DEU/llama2-7b/text_kollektiv',
            output_last_hidden_states=False,
            output_all_hidden_states=False, output_all_activations=False, 
            output_all_pooled_hidden_states=True, output_all_pooled_activations=True)









