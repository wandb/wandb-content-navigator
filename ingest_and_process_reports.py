
import json
import wandb
import pandas as pd
from tqdm import tqdm
from google.cloud import bigquery

def get_reports_from_bigquery(created_after=None):
    client = bigquery.Client(project='wandb-production')

    query = """
    SELECT 
       created_at, 
       description,
       display_name, 
       is_public, 
       created_using,
       report_id, 
       project_id,
       type,
       name,
       report_path, 
       showcased_at, 
       spec, 
       stars_count, 
       user_id, 
       view_count
    FROM 
        analytics.dim_reports
    WHERE
        is_public
        AND showcased_at IS NOT NULL
    """

    if created_after:
        query += f"AND created_at >= '{created_after}'"

    reports_df = client.query(query).to_dataframe()
    return reports_df


def get_fully_connected_spec():
    client = bigquery.Client(project='wandb-production')

    query = """
        SELECT *
        FROM `analytics.stg_mysql_views`
        WHERE type = 'fullyConnected' 
    """

    fc_spec = client.query(query).to_dataframe()
    return fc_spec


# def convert_block_to_markdown(block):
#     """
#     Converts a single content block to its Markdown representation.
#     """
#     md_content = ""

#     # Handle different types of blocks
#     if block['type'] == 'paragraph':
#         for child in block['children']:
#             if 'url' in child:
#                 md_content += f"[{child['children'][0]['text']}]({child['url']})"
#             elif 'inlineCode' in child and child['inlineCode']:
#                 md_content += f"`{child.get('text', '')}`"
#             else:
#                 md_content += child.get('text', '')
#         md_content += "\n\n"

#     # elif block['type'] == 'heading':
#     #     md_content += "#" * block['level'] + " " + ''.join(child['text'] for child in block['children']) + "\n\n"

#     elif block['type'] == 'heading':
#         md_content += "#" * block['level'] + " "
#         for child in block['children']:
#             if 'url' in child:
#                 md_content += f"[{child['children'][0]['text']}]({child['url']})"
#             else:
#                 md_content += child.get('text', '')
#         md_content += "\n\n"

#     # elif block['type'] == 'list':
#     #     for item in block['children']:
#     #         if item['type'] == 'list-item':
#     #             md_content += "* "
#     #             for child in item['children']:
#     #                 if child['type'] == 'paragraph':
#     #                     for text_block in child['children']:
#     #                         if 'inlineCode' in text_block and text_block['inlineCode']:
#     #                             md_content += f"`{text_block.get('text', '')}`"
#     #                         else:
#     #                             md_content += text_block.get('text', '')
#     #             md_content += "\n"
#     #     md_content += "\n"
#     elif block['type'] == 'list':
#         for item in block['children']:
#             if item['type'] == 'list-item':
#                 md_content += "* "
#                 for child in item['children']:
#                     if 'inlineCode' in child and child['inlineCode']:
#                         md_content += f"`{child.get('text', '')}`"
#                     else:
#                         md_content += child.get('text', '')
#                 md_content += "\n"
#         md_content += "\n"

#     elif block['type'] == 'code-block':
#         md_content += "```\n"
#         for line in block['children']:
#             md_content += line['children'][0].get('text', '') + "\n"
#         md_content += "```\n\n"

#     elif block['type'] == 'block-quote' or block['type'] == 'callout-block':
#         md_content += "\n> "
#         for child in block['children']:
#             md_content += child.get('text', '') + " "
#         md_content += "\n\n"

#     elif block['type'] == 'horizontal-rule':
#         md_content += "\n---\n"

#     elif block['type'] == 'latex':
#         latex_content = ''.join(child.get('text', '') for child in block['children'])
#         md_content += f"$$\n{latex_content}\n$$\n\n"

#     # Other specific block types are ignored as per instructions
#     return md_content


def convert_block_to_markdown(block):
    """
    Converts a single content block to its Markdown representation.
    """
    md_content = ""

    # Handle different types of blocks
    if block['type'] == 'paragraph':
        for child in block['children']:
            if 'url' in child:
                md_content += f"[{child['children'][0]['text']}]({child['url']})"
            elif 'inlineCode' in child and child['inlineCode']:
                md_content += f"`{child.get('text', '')}`"
            else:
                md_content += child.get('text', '')
        md_content += "\n\n"

    elif block['type'] == 'heading':
        md_content += "#" * block['level'] + " "
        for child in block['children']:
            if 'url' in child:
                md_content += f"[{child['children'][0]['text']}]({child['url']})"
            else:
                md_content += child.get('text', '')
        md_content += "\n\n"

    elif block['type'] == 'list':
        for item in block['children']:
            if item['type'] == 'list-item':
                md_content += "* "
                for child in item['children']:
                    if child.get('type') == 'paragraph':
                        for text_block in child['children']:
                            if 'inlineCode' in text_block and text_block['inlineCode']:
                                md_content += f"`{text_block.get('text', '')}`"
                            else:
                                md_content += text_block.get('text', '')
                    else:
                        if 'inlineCode' in child and child['inlineCode']:
                            md_content += f"`{child.get('text', '')}`"
                        else:
                            md_content += child.get('text', '')
                md_content += "\n"
        md_content += "\n"

    elif block['type'] == 'code-block':
        md_content += "```\n"
        for line in block['children']:
            md_content += line['children'][0].get('text', '') + "\n"
        md_content += "```\n\n"

    elif block['type'] == 'block-quote' or block['type'] == 'callout-block':
        md_content += "\n> "
        for child in block['children']:
            md_content += child.get('text', '') + " "
        md_content += "\n\n"

    elif block['type'] == 'horizontal-rule':
        md_content += "\n---\n"

    elif block['type'] == 'latex':
        latex_content = block.get('content', '')
        md_content += f"$$\n{latex_content}\n$$\n\n"

    return md_content


def extract_panel_group_text(spec_dict):
    # Initialize an empty list to store the content
    content = ""

    # Extract content under "panelGroups"
    for block in spec_dict.get('panelGroups', []):
        # content_list.append(block.get('content', ''))
        content += block.get('content', '') + "\n"

    # # Extract content under "global" (not sure if we need this)
    # global_block = spec_dict.get('global', {})
    # content_list.append(global_block.get('name', ''))

    return content


def spec_to_markdown(spec, report_id, report_name):
    spec_dict = json.loads(spec)
    markdown_text = ""
    buggy_report_id = None
    spec_type = ""
    is_buggy = False

    if "panelGroups" in spec:
        markdown_text = extract_panel_group_text(spec_dict)
        spec_type = "panelgroup"

    else:
        try:
            for i,block in enumerate(spec_dict["blocks"]):
                try:
                    markdown_text += convert_block_to_markdown(block)
                    spec_type = "blocks"
                except Exception as e:
                    # print(f"Error converting block {i} in report_id: {report_id}, {report_name}:\n{block[:500]}\n{e}\nSPEC DICT:\n{spec_dict}\n")
                    print(f"Error converting block {i} in report_id: {report_id}, {report_name}:\n{e}\nSPEC DICT:\n{spec_dict}\n")

                    markdown_text = ""
                    buggy_report_id = report_id
                    is_buggy = True
        except Exception as e:
            print(f"Error finding 'blocks' in spec for report_id: {report_id}, {report_name} :\n{e}\n")
            markdown_text = ""
            buggy_report_id = report_id
            is_buggy = True
    return markdown_text, buggy_report_id, is_buggy, spec_type


def extract_fc_report_ids(fc_spec_df):
    fc_spec = json.loads(fc_spec_df["spec"].values[0])
    reports_metadata = fc_spec['reportIDsWithTagV2IDs']
    df = pd.json_normalize(
        reports_metadata, 
        'tagIDs', 
        ["id", "authors"]
    ).rename(
        columns = {0: 'tagID', 'id': 'reportID'}
    )

     # Drop duplicates on 'reportID' column
    df = df.drop_duplicates(subset='reportID')
    df['reportID'] = df['reportID'].str.lower()
    # drop the tagID column
    df = df.drop(columns=['tagID'])
    # convert authors to string, unpack it if its a list
    df['authors'] = df['authors'].astype(str)
    return df



####################################


if __name__ == "__main__":
    
    OUTPUT_FILE = 'reports_final.csv'

    ### Download fully connected spec from BigQuery
    # fc_spec_df = get_fully_connected_spec()
    # fc_spec_df.to_csv('fc_spec.csv')

    fc_spec_df = pd.read_csv('fc_spec.csv', index_col=0)
    fc_ids_df = extract_fc_report_ids(fc_spec_df)


    ### Download reports data from BigQuery
    # reports_df = get_reports_from_bigquery()
    # print(f"Downloaded {len(reports_df)} reports")
    # reports_df.to_csv('report_data.csv')

    ### load reports data
    reports_df = pd.read_csv('report_data.csv', index_col=0)
    reports_df['reportID'] = reports_df['report_path'].str.split('reports/--', n=1, expand=True)[1].str.lower()
    # print(reports_df['reportID'].values[:10])


    ### Filter reports down to just Fully Connected reports
    print(f"Before filtering, there are {len(reports_df)} reports")
    reports_df = fc_ids_df.merge(
            reports_df, 
            how='inner', 
            on='reportID', 
            # right_index=True
        )
    print(f"After filtering, there are {len(reports_df)} reports")

    ### Process reports text to markdown
    markdown_ls = []
    buggy_report_ids = []
    spec_type_ls = []
    is_buggy_ls = []
    is_short_report_ls = []
    for _, row in tqdm(reports_df.iterrows()):
        markdown, buggy_report_id, is_buggy, spec_type = spec_to_markdown(row['spec'], 
                                                    row['report_id'],
                                                    row['display_name']
                                                    )
        
        markdown_ls.append(markdown)
        buggy_report_ids.append(buggy_report_id)
        spec_type_ls.append(spec_type)
        is_buggy_ls.append(is_buggy)

        # check if markdown has less than 100 characters
        if len(markdown) < 100:
            is_short_report_ls.append(True)
        else:
            is_short_report_ls.append(False)

    reports_df['markdown_text'] = markdown_ls
    reports_df['spec_type'] = spec_type_ls
    reports_df['is_buggy'] = is_buggy_ls
    reports_df['is_short_report'] = is_short_report_ls

    
    reports_df["content"] = "\n# " + reports_df["display_name"] + "\n\nDescription: " + reports_df["description"] + "\n\nBody:\n\n" + reports_df['markdown_text']
    reports_df["source"] = "https://wandb.ai" + reports_df["report_path"]

    ### Save reports to CSV and W&B

    reports_df.to_csv(OUTPUT_FILE)

    ### Push to W&B 
    wandb.init(name="upload_fully_connected_reports",
               project='wandbot_public',
               entity='wandbot',
               job_type='upload_fc_reports')
    tbl = wandb.Table(dataframe=reports_df)

    artifact = wandb.Artifact('fc-markdown-reports', type='fully-connected-dataset')
    artifact.add_file(OUTPUT_FILE) #, 'fc-markdown-reports')
    wandb.log_artifact(artifact)
    wandb.log({"reports": tbl})
