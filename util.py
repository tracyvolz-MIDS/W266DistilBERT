## The code in this file code in this file was found in the Medium blog post titled, 
## Question Answering with DistilBERT 
## (https://medium.com/@sabrinaherbst/question-answering-with-distilbert-ba3e178fdf3d)

import re
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader 


def normalize_text(s):
    """
    Removes articles and punctuation, and standardizing whitespace are all typical text processing steps.
    Copied from: https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#Metrics-for-QA
    :param s: string to clean
    :return: cleaned string
    """
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    """
    Returns true if the predicted is an exact match, else False
    Retrieved from: https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#Metrics-for-QA
    :param prediction: predicted answer
    :param truth: ground truth
    :return: 1 if exact match, else 0
    """
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    """
    Computes the F-1 score of a prediction, based on the tokens
    Retrieved from: https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#Metrics-for-QA
    :param prediction: predicted answer
    :param truth: ground truth
    :return: the f-1 score of the prediction
    """
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    # get tokens that are in the prediction and gt
    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    # calculate precision and recall
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)

def eval_test_set(model, tokenizer, test_loader, device):
    """
    Calculates the mean EM and mean F-1 score on the test set
    :param model: pytorch model
    :param tokenizer: tokenizer used to encode the samples
    :param test_loader: dataloader object with test data
    :param device: device the model is on
    """
    mean_em = []
    mean_f1 = []
    model.to(device)
    model.eval()
    for batch in tqdm(test_loader):
        # get test data and transfer to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start = batch['start_positions'].to(device)
        end = batch['end_positions'].to(device)

        # predict
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start, end_positions=end)

        # iterate over samples, calculate EM and F-1 for all
        for input_i, s, e, trues, truee in zip(input_ids, outputs['start_logits'], outputs['end_logits'], start, end):
            # get predicted start and end logits (maximum score)
            start_logits = torch.argmax(s)
            end_logits = torch.argmax(e)

            # get predicted answer as string
            ans_tokens = input_i[start_logits: end_logits + 1]
            answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
            predicted = tokenizer.convert_tokens_to_string(answer_tokens)

            # get ground truth as string
            ans_tokens = input_i[trues: truee + 1]
            answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
            true = tokenizer.convert_tokens_to_string(answer_tokens)

            # compute score
            em_score = compute_exact_match(predicted, true)
            f1_score = compute_f1(predicted, true)
            mean_em.append(em_score)
            mean_f1.append(f1_score)
    print("Mean EM: ", np.mean(mean_em))
    print("Mean F-1: ", np.mean(mean_f1))

def compute_exact_match_pure(prediction, truth):
    """
    Returns true if the predicted is an exact match, else False, using raw text
    :param prediction: predicted answer
    :param truth: ground truth
    :return: 1 if exact match, else 0
    """
    return int(prediction == truth)

def compute_f1_pure(prediction, truth):
    """
    Computes the F-1 score of a prediction based on raw text tokens
    :param prediction: predicted answer
    :param truth: ground truth
    :return: the f-1 score of the prediction
    """
    pred_tokens = prediction.split()
    truth_tokens = truth.split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    # get tokens that are in the prediction and gt
    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    # calculate precision and recall
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)

def eval_test_set_pure(model, tokenizer, test_loader, device):
   """
   Calculates the mean EM and mean F-1 score on the test set using pure scoring functions
   :param model: pytorch model
   :param tokenizer: tokenizer used to encode the samples
   :param test_loader: dataloader object with test data
   :param device: device the model is on
   """
   mean_em = []
   mean_f1 = []
   model.to(device)
   model.eval()
   for batch in tqdm(test_loader):
       # get test data and transfer to device
       input_ids = batch['input_ids'].to(device)
       attention_mask = batch['attention_mask'].to(device)
       start = batch['start_positions'].to(device)
       end = batch['end_positions'].to(device)

       # predict
       outputs = model(input_ids, attention_mask=attention_mask, start_positions=start, end_positions=end)

       # iterate over samples, calculate EM and F-1 for all
       for input_i, s, e, trues, truee in zip(input_ids, outputs['start_logits'], outputs['end_logits'], start, end):
           # get predicted start and end logits (maximum score)
           start_logits = torch.argmax(s)
           end_logits = torch.argmax(e)

           # get predicted answer as string, preserving original form
           ans_tokens = input_i[start_logits: end_logits + 1]
           answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
           predicted = tokenizer.convert_tokens_to_string(answer_tokens)

           # get ground truth as string, preserving original form
           ans_tokens = input_i[trues: truee + 1]
           answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
           true = tokenizer.convert_tokens_to_string(answer_tokens)

           # compute score using pure versions
           em_score = compute_exact_match_pure(predicted, true)
           f1_score = compute_f1_pure(predicted, true)
           mean_em.append(em_score)
           mean_f1.append(f1_score)
           
   print("Mean Pure EM: ", np.mean(mean_em))
   print("Mean Pure F-1: ", np.mean(mean_f1))


def count_parameters(model):
    """
    This function prints statistic regarding the trainable parameters
    :param model: pytorch model
    :return: parameters to be fine-tuned
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def print_test_set_incorrect_predictions(model, tokenizer, test_loader, device, num_examples=5):
    """
    Analyze and print examples where the model's predictions are incorrect on the test set.
    """
    print("\nAnalyzing Test Set Incorrect Predictions")
    print("=" * 80)

    model.eval()
    incorrect_examples = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, 
                          attention_mask=attention_mask,
                          start_positions=start_positions,
                          end_positions=end_positions)
            
            # Move tensors to CPU for processing
            input_ids = input_ids.cpu()
            start_positions = start_positions.cpu()
            end_positions = end_positions.cpu()
            start_logits = outputs['start_logits'].cpu()
            end_logits = outputs['end_logits'].cpu()

            for i, (input_i, s, e, trues, truee) in enumerate(zip(input_ids,
                                                                 start_logits,
                                                                 end_logits,
                                                                 start_positions,
                                                                 end_positions)):
                # Get predicted start and end using argmax
                pred_start = torch.argmax(s).item()
                pred_end = torch.argmax(e).item()

                # Ensure end isn't before start
                if pred_end < pred_start:
                    pred_end = pred_start

                # Get predicted answer
                pred_tokens = input_i[pred_start:pred_end + 1]
                answer_tokens = tokenizer.convert_ids_to_tokens(pred_tokens, skip_special_tokens=True)
                pred_answer = tokenizer.convert_tokens_to_string(answer_tokens)

                # Get true answer
                true_tokens = input_i[trues:truee + 1]
                answer_tokens = tokenizer.convert_ids_to_tokens(true_tokens, skip_special_tokens=True)
                true_answer = tokenizer.convert_tokens_to_string(answer_tokens)

                if not true_answer.strip() or not pred_answer.strip():
                    continue

                if normalize_text(true_answer) != normalize_text(pred_answer):
                    context = tokenizer.decode(input_i, skip_special_tokens=True)

                    example_dict = {
                        'context': context,
                        'true_answer': true_answer,
                        'pred_answer': pred_answer,
                        'true_span': (trues.item(), truee.item()),
                        'pred_span': (pred_start, pred_end)
                    }

                    incorrect_examples.append(example_dict)

                    if len(incorrect_examples) >= num_examples:
                        break

            if len(incorrect_examples) >= num_examples:
                break

    # Print results (unchanged)
    print(f"\nFound {len(incorrect_examples)} incorrect test set predictions:")
    print("=" * 80)

    for idx, example in enumerate(incorrect_examples, 1):
        print(f"\nTest Example {idx}:")
        print("-" * 40)
        print("\nContext:")
        print(example['context'])
        print("\nTrue Answer:", example['true_answer'])
        print("Normalized True Answer:", normalize_text(example['true_answer']))
        print("Predicted Answer:", example['pred_answer'])
        print("Normalized Predicted Answer:", normalize_text(example['pred_answer']))
        print("\nAnswer Span Positions:")
        print(f"True span: {example['true_span']}")
        print(f"Predicted span: {example['pred_span']}")
        print("=" * 80)

def analyze_test_set_performance(model, tokenizer, test_loader, device):
    """
    Comprehensive test set analysis with proper device handling
    """
    print("\nBeginning Enhanced Test Set Performance Analysis")
    print("=" * 80)

    model.eval()
    total = 0
    em_scores = []
    f1_scores = []
    span_length_diff = []
    position_diff = []
    error_lengths = []
    
    question_type_stats = {
        'what': {'total': 0, 'incorrect': 0},
        'why': {'total': 0, 'incorrect': 0},
        'when': {'total': 0, 'incorrect': 0},
        'where': {'total': 0, 'incorrect': 0},
        'who': {'total': 0, 'incorrect': 0},
        'how': {'total': 0, 'incorrect': 0},
        'other': {'total': 0, 'incorrect': 0}
    }

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start = batch['start_positions'].to(device)
            end = batch['end_positions'].to(device)
            questions = batch.get('question', None)

            outputs = model(input_ids, attention_mask=attention_mask, 
                          start_positions=start, end_positions=end)
            
            # Move tensors to CPU for processing
            input_ids = input_ids.cpu()
            start = start.cpu()
            end = end.cpu()
            start_logits = outputs['start_logits'].cpu()
            end_logits = outputs['end_logits'].cpu()

            for i, (input_i, s, e, trues, truee) in enumerate(zip(input_ids, 
                                                                 start_logits, 
                                                                 end_logits, 
                                                                 start, end)):
                total += 1

                # Get predicted start and end
                start_logits = torch.argmax(s).item()
                end_logits = torch.argmax(e).item()

                # Get predicted answer
                ans_tokens = input_i[start_logits:end_logits + 1]
                answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
                predicted = tokenizer.convert_tokens_to_string(answer_tokens)

                # Get ground truth
                ans_tokens = input_i[trues:truee + 1]
                answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
                true = tokenizer.convert_tokens_to_string(answer_tokens)

                # Calculate scores
                em_score = compute_exact_match(predicted, true)
                f1_score = compute_f1(predicted, true)
                em_scores.append(em_score)
                f1_scores.append(f1_score)

                if predicted.strip() and true.strip():
                    span_length_diff.append(abs((end_logits - start_logits) - (truee - trues)))
                    position_diff.append(abs(start_logits - trues))
                    
                    if em_score == 0:
                        error_lengths.append(len(predicted.split()))

                if questions is not None:
                    question = questions[i].lower() if isinstance(questions[i], str) else ""
                    q_type = 'other'
                    for qt in ['what', 'why', 'when', 'where', 'who', 'how']:
                        if qt in question:
                            q_type = qt
                            break
                    
                    question_type_stats[q_type]['total'] += 1
                    if em_score == 0:
                        question_type_stats[q_type]['incorrect'] += 1

    # Calculate and return statistics (unchanged)
    stats = {
        'total_examples': total,
        'mean_em': np.mean(em_scores) * 100,
        'mean_f1': np.mean(f1_scores) * 100,
        'avg_span_diff': np.mean(span_length_diff) if span_length_diff else 0,
        'avg_position_diff': np.mean(position_diff) if position_diff else 0,
        'avg_error_length': np.mean(error_lengths) if error_lengths else 0,
        'question_type_stats': question_type_stats
    }

    # Print statistics (unchanged)
    print("\nTest Set Performance Metrics:")
    print("-" * 40)
    print(f"Total test examples evaluated: {stats['total_examples']}")
    print(f"Mean EM Score: {stats['mean_em']:.2f}%")
    print(f"Mean F1 Score: {stats['mean_f1']:.2f}%")
    print(f"Average span length difference: {stats['avg_span_diff']:.2f} tokens")
    print(f"Average position difference: {stats['avg_position_diff']:.2f} tokens")
    print(f"Average length of incorrect answers: {stats['avg_error_length']:.2f} words")

    if any(stats['total'] > 0 for stats in question_type_stats.values()):
        print("\nQuestion Type Analysis:")
        print("-" * 40)
        for q_type, type_stats in question_type_stats.items():
            if type_stats['total'] > 0:
                error_rate = (type_stats['incorrect'] / type_stats['total']) * 100
                print(f"{q_type.upper()} questions:")
                print(f"  Total: {type_stats['total']}")
                print(f"  Incorrect: {type_stats['incorrect']}")
                print(f"  Error Rate: {error_rate:.2f}%")

    return stats

def eval_test_set_by_category(model, tokenizer, test_loader, device):
    """
    Calculates EM and F-1 scores grouped by question type (who, what, when, etc.)
    Returns both overall metrics and breakdown by question type
    """
    category_metrics = {}
    
    model.to(device)
    model.eval()
    
    def get_question_type(text):
        """Extract the question word from the input text"""
        # Convert tokens to text if needed
        if isinstance(text, torch.Tensor):
            text = tokenizer.decode(text, skip_special_tokens=True)
            
        question_words = ['who', 'what', 'when', 'where', 'why', 'how', 'which']
        words = text.lower().split()
        
        for word in words:
            if word in question_words:
                return word
        return 'other'
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start = batch['start_positions'].to(device)
            end = batch['end_positions'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, 
                          start_positions=start, end_positions=end)
            
            # Move tensors to CPU for processing
            input_ids = input_ids.cpu()
            start_logits = outputs['start_logits'].cpu()
            end_logits = outputs['end_logits'].cpu()
            start = start.cpu()
            end = end.cpu()
            
            # Process each sample in the batch
            for input_i, s, e, trues, truee in zip(input_ids, start_logits, 
                                                  end_logits, start, end):
                # Get predicted answer
                start_pred = torch.argmax(s).item()
                end_pred = torch.argmax(e).item()
                
                # Ensure end isn't before start
                if end_pred < start_pred:
                    end_pred = start_pred
                
                ans_tokens = input_i[start_pred:end_pred + 1]
                pred_text = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
                )
                
                # Get ground truth
                true_tokens = input_i[trues:truee + 1]
                true_text = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(true_tokens, skip_special_tokens=True)
                )
                
                # Get question type
                q_type = get_question_type(tokenizer.decode(input_i, skip_special_tokens=True))
                
                # Initialize metrics for new question type
                if q_type not in category_metrics:
                    category_metrics[q_type] = {
                        'em_scores': [],
                        'f1_scores': [],
                        'count': 0
                    }
                
                # Calculate and store metrics
                em_score = compute_exact_match(pred_text, true_text)
                f1_score = compute_f1(pred_text, true_text)
                category_metrics[q_type]['em_scores'].append(em_score)
                category_metrics[q_type]['f1_scores'].append(f1_score)
                category_metrics[q_type]['count'] += 1
    
    # Calculate summary statistics
    summary = {}
    total_samples = sum(m['count'] for m in category_metrics.values())
    
    for q_type, metrics in category_metrics.items():
        summary[q_type] = {
            'count': metrics['count'],
            'percentage': (metrics['count'] / total_samples) * 100,
            'mean_em': np.mean(metrics['em_scores']),
            'mean_f1': np.mean(metrics['f1_scores'])
        }
    
    # Create formatted output
    pt = PrettyTable()
    pt.field_names = ['Question Type', 'Count', '% of Total', 'Mean EM', 'Mean F1']
    
    for q_type, stats in summary.items():
        pt.add_row([
            q_type,
            stats['count'],
            f"{stats['percentage']:.1f}%",
            f"{stats['mean_em']:.3f}",
            f"{stats['mean_f1']:.3f}"
        ])
    
    print(pt)
    return summary

# Display Category Examples
def display_category_examples(model, tokenizer, test_loader, device):
   """
   Collects and displays example successes, failures, and partial matches for each question category
   """
   category_examples = {}

   model.to(device)
   model.eval()

   def get_question_type(text):
       """Extract the question word from the input text"""
       if isinstance(text, torch.Tensor):
           text = tokenizer.decode(text, skip_special_tokens=True)

       question_words = ['who', 'what', 'when', 'where', 'why', 'how', 'which']
       words = text.lower().split()

       for word in words:
           if word in question_words:
               return word
       return 'other'

   with torch.no_grad():
       for batch in tqdm(test_loader):
           input_ids = batch['input_ids'].to(device)
           attention_mask = batch['attention_mask'].to(device)
           start = batch['start_positions'].to(device)
           end = batch['end_positions'].to(device)

           outputs = model(input_ids, attention_mask=attention_mask,
                         start_positions=start, end_positions=end)

           # Move tensors to CPU for processing
           input_ids = input_ids.cpu()
           start_logits = outputs['start_logits'].cpu()
           end_logits = outputs['end_logits'].cpu()
           start = start.cpu()
           end = end.cpu()

           # Process each sample in the batch
           for input_i, s, e, trues, truee in zip(input_ids, start_logits,
                                                 end_logits, start, end):
               # Get full question text
               full_text = tokenizer.decode(input_i, skip_special_tokens=True)

               # Get predicted answer
               start_pred = torch.argmax(s).item()
               end_pred = torch.argmax(e).item()

               # Ensure end isn't before start
               if end_pred < start_pred:
                   end_pred = start_pred

               ans_tokens = input_i[start_pred:end_pred + 1]
               pred_text = tokenizer.convert_tokens_to_string(
                   tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
               )

               # Get ground truth
               true_tokens = input_i[trues:truee + 1]
               true_text = tokenizer.convert_tokens_to_string(
                   tokenizer.convert_ids_to_tokens(true_tokens, skip_special_tokens=True)
               )

               # Get question type
               q_type = get_question_type(full_text)

               # Initialize examples for new question type
               if q_type not in category_examples:
                   category_examples[q_type] = {
                       'successes': [],      # Store successful examples (EM=1)
                       'failures': [],       # Store failed examples (EM=0)
                       'partial_matches': [] # Store examples with F1 between 0.3-0.7
                   }

               # Calculate metrics
               em_score = compute_exact_match(pred_text, true_text)
               f1_score = compute_f1(pred_text, true_text)

               example = {
                   'question': full_text,
                   'predicted': pred_text,
                   'truth': true_text,
                   'f1_score': f1_score
               }

               # Store based on performance
               if em_score == 1:
                   if len(category_examples[q_type]['successes']) < 5:
                       category_examples[q_type]['successes'].append(example)
               elif 0.3 <= f1_score <= 0.7:
                   if len(category_examples[q_type]['partial_matches']) < 5:
                       category_examples[q_type]['partial_matches'].append(example)
               else:
                   if len(category_examples[q_type]['failures']) < 5:
                       category_examples[q_type]['failures'].append(example)

   # Display examples for each category
   for q_type, examples in category_examples.items():
       print(f"\n{'='*20} {q_type.upper()} Questions {'='*20}")

       print("\n✅ Successful Examples (EM=1):")
       for i, example in enumerate(examples['successes'], 1):
           print(f"\n{i}. Question: {example['question']}")
           print(f"   Predicted: {example['predicted']}")
           print(f"   Truth: {example['truth']}")

       print("\n⚠️ Partial Matches (0.3 ≤ F1 ≤ 0.7):")
       for i, example in enumerate(examples['partial_matches'], 1):
           print(f"\n{i}. Question: {example['question']}")
           print(f"   Predicted: {example['predicted']}")
           print(f"   Truth: {example['truth']}")
           print(f"   F1 Score: {example['f1_score']:.3f}")

       print("\n❌ Failed Examples (EM=0, F1 < 0.3 or F1 > 0.7):")
       for i, example in enumerate(examples['failures'], 1):
           print(f"\n{i}. Question: {example['question']}")
           print(f"   Predicted: {example['predicted']}")
           print(f"   Truth: {example['truth']}")

   return category_examples
