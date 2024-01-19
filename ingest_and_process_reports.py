import json
import wandb
import argparse
import pandas as pd
from tqdm import tqdm
from google.cloud import bigquery


def get_reports_ids():
    client = bigquery.Client(project='wandb-production')

    query = """
        SELECT DISTINCT
            lower(REGEXP_REPLACE(COALESCE(
            REGEXP_EXTRACT(report_path, "(--[Vv]ml[^/?]+)"),
            REGEXP_EXTRACT(report_path, "(/[Vv]ml[^/?]+)"), 
            REGEXP_EXTRACT(report_path, "([Vv]ml[^/?]+)")),
            "^[/-]+", "")) as reportID
        FROM 
            analytics.dim_reports
        WHERE
            is_public
        """
    
    report_ids_df = client.query(query).to_dataframe()
    return report_ids_df


def get_reports_from_bigquery(report_ids : str, created_after=None):
    
    client = bigquery.Client(project='wandb-production')

    query = f"""
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
        and LOWER(REGEXP_EXTRACT(report_path, r'reports/--(.*)')) in ({report_ids})
    """

    # # AND showcased_at IS NOT NULL

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


def convert_block_to_markdown(block):
    """
    Converts a single content block to its Markdown representation.
    """
    md_content = ""

    # Handle different types of blocks
    if block['type'] == 'paragraph':
        for child in block['children']:
            if 'type' in child:
                if child['type'] == 'block-quote' or child['type'] == 'callout-block':
                    md_content += convert_block_to_markdown(child)
                elif 'url' in child:
                    md_content += f"[{child['children'][0]['text']}]({child['url']})"
                elif 'inlineCode' in child and child['inlineCode']:
                    md_content += f"`{child.get('text', '')}`"
                else:
                    md_content += child.get('text', '')
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



def main(args):
    OUTPUT_FILE = 'reports_final.jsonl'

    ### Download fully connected spec from BigQuery
    
    if args.refresh_data:
        # Get all public reports ids from BigQuery
        report_ids_df = get_reports_ids()
        report_ids_df.to_csv('all_report_ids.csv')

        # Get views info for just Fully Connected reports from BigQuery
        fc_spec_df = get_fully_connected_spec()
        fc_spec_df.to_csv('fc_spec.csv')

        fc_ids_df = extract_fc_report_ids(fc_spec_df)


        ### Filter report IDs down to just Fully Connected reports using the IDs from fc_ids_df
        print(f"Before filtering, there are {len(report_ids_df)} reports")

        report_ids_df = report_ids_df.merge(
                fc_ids_df, 
                how='inner', 
                on='reportID', 
            )
        
        print(f"After filtering, there are {len(report_ids_df)} FC report IDs to fetch")
        
        # Pus report ids into a string for BigQuery query
        report_ids_str = ""
        for id in report_ids_df['reportID'].values:
            report_ids_str += f"'{id}',"


        ### Get all reports data from BigQuery
        ### load reports data
        # reports_df = pd.read_csv('report_data.csv', index_col=0)
        # reports_df['reportID'] = reports_df['report_path'].str.split('reports/--', n=1, expand=True)[1].str.lower()
        # # print(reports_df['reportID'].values[:10])

        ### Get just Fully Connected reports data from BigQuery
        reports_df = get_reports_from_bigquery(report_ids_str[:-1])
        reports_df["source"] = "https://wandb.ai" + reports_df["report_path"]
        reports_df['reportID'] = reports_df['report_path'].str.split('reports/--', n=1, expand=True)[1].str.lower()
        reports_df['description'] = reports_df['description'].fillna('')
        reports_df['display_name'] = reports_df['display_name'].fillna('')
        # reports_df.to_csv('raw_reports_data.csv')
        reports_df.to_json('raw_reports_data.jsonl', orient='records', lines=True)
        print(f"{len(reports_df)} Fully Connected Reports fetched")
    
    else:
        reports_df = pd.read_json('raw_reports_data.jsonl', lines=True)
        print(reports_df.head())
        print(reports_df.columns)


    ### Process reports text to markdown
    markdown_ls = []
    buggy_report_ids = []
    spec_type_ls = []
    is_buggy_ls = []
    is_short_report_ls = []
    for idx, row in tqdm(reports_df.iterrows()):
        # if row["source"] == "https://wandb.ai/wandb_fc/gradient-dissent/reports/--VmlldzozMzE0Njgx":
        #     print(row["spec"])
        #     break
        if row['spec'] is None or isinstance(row['spec'], float): 
            print(idx)
            markdown_ls.append("spec-error")
            buggy_report_ids.append(row['report_id'])
            spec_type_ls.append("spec-error")
            is_buggy_ls.append(True)
            is_short_report_ls.append(True)
            
        else:
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

    reports_df["content"] = "\n# " + reports_df["display_name"] + "\n\nDescription: " + reports_df["description"] + "\n\nBody:\n\n" + reports_df['markdown_text'].astype(str)
    
    for ii,uu in enumerate(reports_df["content"].values):
        if isinstance(uu, float):
            print(uu)
            print(f"Content: {reports_df['content'].values[ii]}")
            print(f"Markdown text: {reports_df['markdown_text'].values[ii][:100]}")
            print(f"Description: {reports_df['description'].values[ii]}")
            print(f"Title: {reports_df['display_name'].values[ii]}")
            print()
            import sys
            sys.exit("Exiting the script.")

    reports_df["character_count"] = reports_df["content"].map(len)

    # reports_df["character_count"] = len(reports_df["content"])
    reports_df["source"] = "https://wandb.ai" + reports_df["report_path"]

    # tidy up the dataframe
    reports_df.drop(columns=['markdown_text'], inplace=True)
    reports_df.drop(columns=['spec'], inplace=True)
    reports_df.sort_values(by=['created_at'], inplace=True, ascending=False)

    ### Save reports to CSV and W&B
    # print(reports_df.head())
    # reports_df.to_csv(OUTPUT_FILE)

    reports_df.to_json(OUTPUT_FILE, orient='records', lines=True)

    skip_wandb = True
    if not skip_wandb:
        ### Push to W&B 
        wandb.init(name="upload_fully_connected_reports",
                project='wandbot_public',
                entity='wandbot',
                job_type='upload_fc_reports')
        tbl = wandb.Table(dataframe=reports_df)  # reports_df.iloc[:50,:])

        artifact = wandb.Artifact('fc-markdown-reports', type='fully-connected-dataset')
        artifact.add_file(OUTPUT_FILE) #, 'fc-markdown-reports')
        wandb.log_artifact(artifact)
        wandb.log({"reports4": tbl})


####################################


if __name__ == "__main__":
    print("Starting...")
    parser = argparse.ArgumentParser(description='Get Fully Connected Reports from BigQuery')
    parser.add_argument('--refresh_data', 
        action='store_true', 
        help='refresh data from BigQuery')
    args = parser.parse_args()

    main(args)


    ## Try simple_parsing at some point
    # from dataclasses import dataclass
    # from simple_parsing import ArgumentParser

    # parser = ArgumentParser()
    # parser.add_argument("--foo", type=int, default=123, help="foo help")

    # @dataclass
    # class Options:
    #     """ Help string for this group of command-line arguments """
    #     log_dir: str                # Help string for a required str argument
    #     learning_rate: float = 1e-4 # Help string for a float argument

    # parser.add_arguments(Options, dest="options")

    # args = parser.parse_args()
    # print("foo:", args.foo)
    # print("options:", args.options)
    
    