import os
import email
import random
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from email.parser import Parser
from email import policy
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MultiDatasetCombiner:
    def __init__(self, output_dir='data'):
        """
        Initialize multi-dataset combiner
        
        Args:
            output_dir: Directory to save processed datasets
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_email_content(self, file_path):
        """
        Extract email body from email file
        
        Args:
            file_path: Path to email file
            
        Returns:
            Extracted email text
        """
        try:
            with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                msg = email.message_from_file(f, policy=policy.default)
                
                # Try to get email body
                body = ""
                
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        
                        if content_type == "text/plain" and "attachment" not in content_disposition:
                            try:
                                body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                break
                            except:
                                continue
                else:
                    try:
                        body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except:
                        body = str(msg.get_payload())
                
                # Get subject
                subject = msg.get('Subject', '')
                
                # Combine subject and body
                text = f"{subject} {body}".strip()
                
                # Clean text
                text = self.clean_text(text)
                
                return text
                
        except Exception as e:
            return ""
    
    def clean_text(self, text):
        """
        Clean and normalize email text
        
        Args:
            text: Raw email text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove email headers (lines starting with common header keywords)
        lines = text.split('\n')
        cleaned_lines = []
        skip_headers = {'from:', 'to:', 'subject:', 'date:', 'cc:', 'bcc:', 
                       'received:', 'return-path:', 'x-', 'content-', 'mime-',
                       'message-id:', 'reply-to:', 'errors-to:'}
        
        for line in lines:
            line_lower = line.lower().strip()
            if not any(line_lower.startswith(h) for h in skip_headers):
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\@\$\%]', '', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def load_enron_emails(self, enron_path='datasets/enron/maildir', max_emails=10000):
        """
        Load Enron emails (legitimate emails)
        
        Args:
            enron_path: Path to Enron maildir
            max_emails: Maximum number of emails to load
            
        Returns:
            List of email texts
        """
        print("\n📧 Loading Enron Dataset (Ham emails)...")
        
        if not os.path.exists(enron_path):
            print(f"⚠️  Warning: Enron dataset not found at {enron_path}")
            print(f"   Skipping Enron dataset...")
            return []
        
        emails = []
        email_files = []
        
        # Find all email files
        for root, dirs, files in os.walk(enron_path):
            for file in files:
                if not file.startswith('.'):
                    email_files.append(os.path.join(root, file))
                    if len(email_files) >= max_emails:
                        break
            if len(email_files) >= max_emails:
                break
        
        print(f"   Found {len(email_files)} email files")
        
        # Process emails with progress bar
        for file_path in tqdm(email_files, desc="   Processing Enron"):
            text = self.extract_email_content(file_path)
            if text and len(text) > 50:
                emails.append(text)
        
        print(f"   ✓ Loaded {len(emails)} valid Enron emails")
        return emails
    
    def load_spamassassin_emails(self, spamassassin_path='datasets/spamassassin'):
        """
        Load SpamAssassin emails (both spam and ham)
        
        Args:
            spamassassin_path: Path to SpamAssassin dataset
            
        Returns:
            Tuple of (spam_emails, ham_emails)
        """
        print("\n🚫 Loading SpamAssassin Dataset...")
        
        if not os.path.exists(spamassassin_path):
            print(f"⚠️  Warning: SpamAssassin dataset not found at {spamassassin_path}")
            print(f"   Skipping SpamAssassin dataset...")
            return [], []
        
        spam_emails = []
        ham_emails = []
        
        # Define folders for each category
        spam_folders = ['spam', 'spam_2']
        ham_folders = ['easy_ham', 'hard_ham', 'easy_ham_2']
        
        # Load spam emails
        print("   Loading spam emails...")
        for folder in spam_folders:
            folder_path = os.path.join(spamassassin_path, folder)
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if not f.startswith('.')]
                for file in tqdm(files, desc=f"   {folder}"):
                    file_path = os.path.join(folder_path, file)
                    if os.path.isfile(file_path):
                        text = self.extract_email_content(file_path)
                        if text and len(text) > 50:
                            spam_emails.append(text)
        
        print(f"   ✓ Loaded {len(spam_emails)} spam emails")
        
        # Load ham emails
        print("   Loading ham emails...")
        for folder in ham_folders:
            folder_path = os.path.join(spamassassin_path, folder)
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if not f.startswith('.')]
                for file in tqdm(files, desc=f"   {folder}"):
                    file_path = os.path.join(folder_path, file)
                    if os.path.isfile(file_path):
                        text = self.extract_email_content(file_path)
                        if text and len(text) > 50:
                            ham_emails.append(text)
        
        print(f"   ✓ Loaded {len(ham_emails)} ham emails")
        
        return spam_emails, ham_emails
    
    def load_trec_emails(self, trec_path='datasets/trec'):
        """
        Load TREC 2007 spam dataset
        
        Expected structure:
        datasets/trec/
        ├── data/
        │   ├── inmail.1
        │   ├── inmail.2
        │   └── ...
        └── full/index
        
        Args:
            trec_path: Path to TREC dataset
            
        Returns:
            Tuple of (spam_emails, ham_emails)
        """
        print("\n Loading TREC Dataset...")
        
        if not os.path.exists(trec_path):
            print(f"  Warning: TREC dataset not found at {trec_path}")
            print(f"   Skipping TREC dataset...")
            return [], []
        
        spam_emails = []
        ham_emails = []
        
        # Load index file (contains labels)
        index_file = os.path.join(trec_path, 'full', 'index')
        data_path = os.path.join(trec_path, 'data')
        
        if not os.path.exists(index_file):
            print(f"   Warning: Index file not found at {index_file}")
            print(f"   Trying to load all files as unlabeled...")
            # Load all files without labels
            if os.path.exists(data_path):
                files = [f for f in os.listdir(data_path) if not f.startswith('.')]
                print(f"   Found {len(files)} email files")
                for file in tqdm(files, desc="   Processing TREC"):
                    file_path = os.path.join(data_path, file)
                    if os.path.isfile(file_path):
                        text = self.extract_email_content(file_path)
                        if text and len(text) > 50:
                            ham_emails.append(text)  # Assume ham if no labels
            return [], ham_emails
        
        # Read labels
        labels = {}
        with open(index_file, 'r', encoding='latin-1', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    label = parts[0]  # 'spam' or 'ham'
                    filename = parts[1]
                    labels[filename] = label
        
        print(f"   Found {len(labels)} labeled emails")
        
        # Process emails with labels
        for filename, label in tqdm(labels.items(), desc="   Processing TREC"):
            file_path = os.path.join(data_path, filename)
            if os.path.isfile(file_path):
                text = self.extract_email_content(file_path)
                if text and len(text) > 50:
                    if label.lower() == 'spam':
                        spam_emails.append(text)
                    else:
                        ham_emails.append(text)
        
        print(f"   ✓ Loaded {len(spam_emails)} spam emails")
        print(f"   ✓ Loaded {len(ham_emails)} ham emails")
        
        return spam_emails, ham_emails
    
    def load_lingspam_emails(self, lingspam_path='datasets/lingspam'):
        """
        Load LingSpam dataset
        
        Expected structure:
        datasets/lingspam/
        ├── bare/
        │   ├── part1/
        │   ├── part2/
        │   └── ...
        
        Files starting with 'spm' are spam, others are ham
        
        Args:
            lingspam_path: Path to LingSpam dataset
            
        Returns:
            Tuple of (spam_emails, ham_emails)
        """
        print("\n💬 Loading LingSpam Dataset...")
        
        if not os.path.exists(lingspam_path):
            print(f"   Warning: LingSpam dataset not found at {lingspam_path}")
            print(f"   Skipping LingSpam dataset...")
            return [], []
        
        spam_emails = []
        ham_emails = []
        
        # Look for bare directory (preprocessed emails)
        bare_path = os.path.join(lingspam_path, 'bare')
        if not os.path.exists(bare_path):
            # Try lemm_stop or lemm directories
            for alt in ['lemm_stop', 'lemm', 'stop']:
                alt_path = os.path.join(lingspam_path, alt)
                if os.path.exists(alt_path):
                    bare_path = alt_path
                    break
        
        if not os.path.exists(bare_path):
            print(f"   Warning: No suitable directory found in {lingspam_path}")
            return [], []
        
        # Process all part directories
        parts = [d for d in os.listdir(bare_path) if os.path.isdir(os.path.join(bare_path, d))]
        print(f"   Found {len(parts)} parts")
        
        for part in tqdm(parts, desc="   Processing LingSpam"):
            part_path = os.path.join(bare_path, part)
            files = [f for f in os.listdir(part_path) if not f.startswith('.')]
            
            for file in files:
                file_path = os.path.join(part_path, file)
                if os.path.isfile(file_path):
                    # Read file content directly (already preprocessed)
                    try:
                        with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                            text = f.read().strip()
                        
                        if text and len(text) > 50:
                            # Files starting with 'spm' are spam
                            if file.startswith('spm'):
                                spam_emails.append(text)
                            else:
                                ham_emails.append(text)
                    except:
                        continue
        
        print(f"   ✓ Loaded {len(spam_emails)} spam emails")
        print(f"   ✓ Loaded {len(ham_emails)} ham emails")
        
        return spam_emails, ham_emails
    
    def create_balanced_dataset(self, spam_emails, ham_emails, balance_method='undersample'):
        """
        Create a balanced dataset from spam and ham emails
        
        Args:
            spam_emails: List of spam email texts
            ham_emails: List of ham email texts
            balance_method: 'undersample' or 'keep_all'
            
        Returns:
            pandas DataFrame with balanced dataset
        """
        print("\n  Creating dataset...")
        
        print(f"   Total spam emails: {len(spam_emails)}")
        print(f"   Total ham emails: {len(ham_emails)}")
        
        if balance_method == 'undersample':
            # Balance to minimum count
            min_count = min(len(spam_emails), len(ham_emails))
            print(f"   Balancing to: {min_count} emails per class")
            
            # Sample equal amounts
            random.seed(42)
            if len(spam_emails) > min_count:
                spam_sample = random.sample(spam_emails, min_count)
            else:
                spam_sample = spam_emails
            
            if len(ham_emails) > min_count:
                ham_sample = random.sample(ham_emails, min_count)
            else:
                ham_sample = ham_emails
            
            df = pd.DataFrame({
                'text': list(spam_sample) + list(ham_sample),
                'label': [1] * len(spam_sample) + [0] * len(ham_sample)
            })
        else:
            # Keep all data
            print(f"   Keeping all emails (unbalanced)")
            df = pd.DataFrame({
                'text': spam_emails + ham_emails,
                'label': [1] * len(spam_emails) + [0] * len(ham_emails)
            })
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"   ✓ Created dataset with {len(df)} emails")
        print(f"   Spam: {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.1f}%)")
        print(f"   Ham: {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.1f}%)")
        
        return df
    
    def save_dataset(self, df, filename='multi_dataset_spam.csv'):
        """
        Save dataset to CSV
        
        Args:
            df: pandas DataFrame
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"\n💾 Dataset saved to: {output_path}")
        
        # Print statistics
        print(f"\n Final Dataset Statistics:")
        print(f"   Total emails: {len(df)}")
        print(f"   Spam emails: {sum(df['label'] == 1)}")
        print(f"   Ham emails: {sum(df['label'] == 0)}")
        print(f"   Average text length: {df['text'].str.len().mean():.0f} characters")
        print(f"   Min text length: {df['text'].str.len().min()}")
        print(f"   Max text length: {df['text'].str.len().max()}")
        
        return output_path
    
    def combine_all_datasets(self,
                           enron_path='datasets/enron/maildir',
                           spamassassin_path='datasets/spamassassin',
                           trec_path='datasets/trec',
                           lingspam_path='datasets/lingspam',
                           max_enron_emails=10000,
                           balance_method='undersample'):
        """
        Main function to combine all datasets
        
        Args:
            enron_path: Path to Enron dataset
            spamassassin_path: Path to SpamAssassin dataset
            trec_path: Path to TREC dataset
            lingspam_path: Path to LingSpam dataset
            max_enron_emails: Maximum Enron emails to load
            balance_method: 'undersample' or 'keep_all'
            
        Returns:
            Path to saved dataset
        """
        print("\n" + "="*70)
        print("COMBINING MULTIPLE SPAM DATASETS")
        print("Enron + SpamAssassin + TREC + LingSpam")
        print("="*70)
        
        all_spam = []
        all_ham = []
        
        # Load each dataset
        spam, ham = self.load_spamassassin_emails(spamassassin_path)
        all_spam.extend(spam)
        all_ham.extend(ham)
        
        spam, ham = self.load_trec_emails(trec_path)
        all_spam.extend(spam)
        all_ham.extend(ham)
        
        spam, ham = self.load_lingspam_emails(lingspam_path)
        all_spam.extend(spam)
        all_ham.extend(ham)
        
        # Load Enron (all ham)
        enron_ham = self.load_enron_emails(enron_path, max_enron_emails)
        all_ham.extend(enron_ham)
        
        print(f"\n Combined Statistics:")
        print(f"   Total spam emails: {len(all_spam)}")
        print(f"   Total ham emails: {len(all_ham)}")
        
        # Create dataset
        df = self.create_balanced_dataset(all_spam, all_ham, balance_method)
        
        # Save dataset
        output_path = self.save_dataset(df, 'combined_multi_spam_dataset.csv')
        
        return output_path


def main():
    """Main execution function"""
    combiner = MultiDatasetCombiner(output_dir='data')
    
    print("\n" + "="*70)
    print("MULTI-DATASET SPAM PREPARATION")
    print("="*70)
    print("\nExpected directory structure:")
    print("datasets/")
    print("├── enron/maildir/")
    print("├── spamassassin/")
    print("├── trec/")
    print("└── lingspam/")
    print("\n" + "="*70)
    
    # Combine all datasets
    output_path = combiner.combine_all_datasets(
        enron_path='datasets/enron/maildir',
        spamassassin_path='datasets/spamassassin',
        trec_path='datasets/trec',
        lingspam_path='datasets/lingspam',
        max_enron_emails=10000,
        balance_method='undersample'  # or 'keep_all'
    )
    
    print("\n" + "="*70)
    print("DATASET PREPARATION COMPLETE!")
    print("="*70)
    print(f"\nDataset ready at: {output_path}")
    print("\nNext steps:")
    print("   1. Run: python train/spam_train.py")
    print("   2. Train with LightGBM and TF-IDF")
    print("   3. Deploy your model!")
    print("="*70)


if __name__ == "__main__":
    main()