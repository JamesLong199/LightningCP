# def get_next_indices(interval_seq, i, max_parallel=float('inf'), num_total=-1):
#     # Case 1: i is in interval_seq - standard behavior with max_parallel limit
#     # print('pipeline 43 interval_seq:', interval_seq, flush=True)
#     # print('pipeline 44 i:', i, flush=True)

#     if i in interval_seq:
#         # print('pipeline 700 i in interval_seq:', i, flush=True)
#         current_idx = interval_seq.index(i)
#         # print('pipeline 702 current_idx:', current_idx, flush=True)
#         # Check if we're at the last item
#         if current_idx == len(interval_seq) - 1:
#             # print('pipeline 704 current_idx == len(interval_seq) - 1:', current_idx, flush=True)
#             return []

#         # # If i is after the last interval point, return empty list
#         # if interval_seq and i > interval_seq[-1]:
#         #     assert num_total != -1
#         #     print('pipeline 58 i is after the last interval point:', i, flush=True)
#         #     return list(range(i, min(i + max_parallel, num_total)))
        
#         next_item = interval_seq[current_idx + 1]
#         # print('pipeline 706 next_item:', next_item, flush=True) 
#         # print('pipeline 707 list(range(i + 1, min(i + 1 + max_parallel, next_item))):', list(range(i + 1, min(i + 1 + max_parallel, next_item))), flush=True)
#         # Return list truncated to max_parallel
#         return list(range(i + 1, min(i + 1 + max_parallel, next_item)))
    
#     # Case 2: i is not in interval_seq - find the appropriate interval
#     else:
#         # print('pipeline 69 i is not in interval_seq:', i, flush=True)
#         # Find the enclosing interval
#         for idx in range(len(interval_seq) - 1):
#             current = interval_seq[idx]
#             next_item = interval_seq[idx + 1]
            
#             # Check if i falls between two interval points
#             if current < i < next_item:
#                 # Continue from i to the next_item, limited by max_parallel
#                 return list(range(i, min(i + max_parallel, next_item)))
        
#         # If i is after the last interval point, return empty list
#         if interval_seq and i > interval_seq[-1]:
#             assert num_total != -1
#             # print('pipeline 83 i is after the last interval point:', i, flush=True)
#             return list(range(i, min(i + max_parallel, num_total)))
            
#         # If i is before the first interval point
#         if interval_seq and i < interval_seq[0]:
#             # print('pipeline 88 i is before the first interval point:', i, flush=True)
#             return list(range(i, min(i + max_parallel, interval_seq[0])))
            
#         return []

# num_total = 30
# interval_seq = [0, 3, 6, 9, 12, 24, 27]
# t_i = 27

# # num_total = 6
# # interval_seq = [0, 1, 5]
# # t_i = 2

# # interval_seq = [0, 1, 2]
# # t_i = 3

# # interval_seq = [0, 1, 4]
# # t_i = 5

# if t_i-1 in interval_seq and interval_seq.index(t_i-1) != len(interval_seq)-1:
#     parallel_indices = get_next_indices(interval_seq, t_i-1, max_parallel=4, num_total=30)
# elif t_i not in interval_seq and t_i < num_total-1:
#     parallel_indices = get_next_indices(interval_seq, t_i, max_parallel=4, num_total=30)
# else:
#     parallel_indices = []

# print(parallel_indices)


def get_bg_features_context(c, do_classifier_free_guidance, total_length):
    if not do_classifier_free_guidance:
        return c
    else:
        if total_length-1 in c and 0 in c:
            end_idx = c.index(total_length-1)
            print('end_idx:', end_idx, flush=True)
            if end_idx == 0:
                new_c = [c[0]*2, c[0]*2+1] + list(range(2 * c[1], 2 * c[-1] + 2))
            else:
                
                new_c = list(range(2 * c[0], 2 * c[end_idx] + 2)) + list(range(2 * c[end_idx+1], 2 * c[-1] + 2))
            return new_c
        else:
            return list(range(2 * c[0], 2 * c[-1] + 2))

do_classifier_free_guidance = True
c = [171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 0]
print(get_bg_features_context(c, do_classifier_free_guidance, 182))