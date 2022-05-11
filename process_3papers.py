from utils.parse_paper import parse_paper
from utils.combine_df import combine_df
# from utils.preprocess_data import preprocess_exam_data

def process_3papers():
    '''parse 3 papers from csv, combine papers and preprocess inc. thres, binarise'''

    # Prepare data
    exam_df1, meta_df1 = parse_paper('new1')
    exam_df2, meta_df2 = parse_paper('new2')
    exam_df3, meta_df3 = parse_paper('new3')

    combined_exam_df = combine_df([exam_df1, exam_df2, exam_df3])
    combined_meta_df = combine_df([meta_df1, meta_df2, meta_df3])
    processed_df, processed_meta_df = preprocess_exam_data(combined_exam_df, combined_meta_df)

    return processed_df, processed_meta_df
