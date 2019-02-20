/*******************************************************************************
 * Copyright 2014
 * Ubiquitous Knowledge Processing (UKP) Lab and FG Language Technology
 * Technische Universität Darmstadt
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/
package de.tudarmstadt.ukp.dkpro.core.io.conll;

import static org.apache.commons.io.IOUtils.closeQuietly;
import static org.apache.uima.fit.util.JCasUtil.select;
import static org.apache.uima.fit.util.JCasUtil.selectCovered;

import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.Feature;
import org.apache.uima.cas.Type;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.descriptor.TypeCapability;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.hucompute.services.type.biocreative.Abstract;

import de.tudarmstadt.ukp.dkpro.core.api.io.IobEncoder;
import de.tudarmstadt.ukp.dkpro.core.api.io.JCasFileWriter_ImplBase;
import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity;
import de.tudarmstadt.ukp.dkpro.core.api.parameter.ComponentParameters;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;

/**
 * <p>Writes the CoNLL 2002 named entity format. The columns are separated by a single space, unlike
 * illustrated below.</p>
 * 
 * <pre><code>
 * Wolff      B-PER
 * ,          O
 * currently  O
 * a          O
 * journalist O
 * in         O
 * Argentina  B-LOC
 * ,          O
 * played     O
 * with       O
 * Del        B-PER
 * Bosque     I-PER
 * in         O
 * the        O
 * final      O
 * years      O
 * of         O
 * the        O
 * seventies  O
 * in         O
 * Real       B-ORG
 * Madrid     I-ORG
 * .          O
 * </code></pre>
 * 
 * <ol>
 * <li>FORM - token</li>
 * <li>NER - named entity (BIO encoded)</li>
 * </ol>
 * 
 * <p>Sentences are separated by a blank new line.</p>
 * 
 * @see <a href="http://www.clips.ua.ac.be/conll2002/ner/">CoNLL 2002 shared task</a>
 */
@TypeCapability(inputs = { "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData",
		"de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence",
		"de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token",
"de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"})
public class Conll2002Writer
extends JCasFileWriter_ImplBase
{

	/**
	 * Write token annotations to the CAS.
	 */
	public static final String PARAM_ANNOTATION_FILE = "PARAM_ANNOTATION_FILE";
	@ConfigurationParameter(name = PARAM_ANNOTATION_FILE, mandatory = false)
	private String annotationFile;


	private static final String UNUSED = "_";

	/**
	 * Name of configuration parameter that contains the character encoding used by the input files.
	 */
	public static final String PARAM_ENCODING = ComponentParameters.PARAM_SOURCE_ENCODING;
	@ConfigurationParameter(name = PARAM_ENCODING, mandatory = true, defaultValue = "UTF-8")
	private String encoding;

	public static final String PARAM_FILENAME_SUFFIX = "filenameSuffix";
	@ConfigurationParameter(name = PARAM_FILENAME_SUFFIX, mandatory = true, defaultValue = ".conll")
	private String filenameSuffix;

	public static final String PARAM_SPLIT_AT_SENTENCE = "PARAM_SPLIT_AT_SENTENCE";
	@ConfigurationParameter(name = PARAM_SPLIT_AT_SENTENCE, mandatory = true, defaultValue = "true")
	private boolean splitAtSentence;

	public static final String PARAM_WRITE_NAMED_ENTITY = ComponentParameters.PARAM_WRITE_NAMED_ENTITY;
	@ConfigurationParameter(name = PARAM_WRITE_NAMED_ENTITY, mandatory = true, defaultValue = "true")
	private boolean writeNamedEntity;

	public static final String PARAM_SERPERATOR = "SEPERATOR";
	@ConfigurationParameter(name = PARAM_SERPERATOR, mandatory = true, defaultValue = " ")
	private String seperator;

	/**
	 * Write token annotations to the CAS.
	 */
	public static final String PARAM_FILTER_FILES_ONLY_WITH_ANNOTATIONS = "PARAM_FILTER_FILES_ONLY_WITH_ANNOTATIONS";
	@ConfigurationParameter(name = PARAM_FILTER_FILES_ONLY_WITH_ANNOTATIONS, mandatory = true)
	private boolean filter;

	/**
	 * Write token annotations to the CAS.
	 */
	public static final String PARAM_DOCUMENT_IDS_TO_WRITE = "PARAM_DOCUMENT_IDS_TO_WRITE";
	@ConfigurationParameter(name = PARAM_DOCUMENT_IDS_TO_WRITE, mandatory = false)
	private List<String> ids;


	HashMap<String,ArrayList<String>> annotations = new HashMap<>();
	int docCount = 0;
	@Override
	public void initialize(UimaContext context) throws ResourceInitializationException {
		// TODO Auto-generated method stub
		super.initialize(context);
		try {
			List<String>lines = FileUtils.readLines(new File(annotationFile));
			for (String string : lines) {
				String id = string.split("\t")[0];
				if(!annotations.containsKey(id))
					annotations.put(id, new ArrayList<>());
				annotations.get(id).add(string);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}


	@Override
	public void process(JCas aJCas)
			throws AnalysisEngineProcessException
	{
		if(aJCas.getDocumentText()==null)
			return;
		PrintWriter out = null;
		try {
			out = new PrintWriter(new OutputStreamWriter(getOutputStream(aJCas, filenameSuffix),
					encoding));
			convert(aJCas, out);
		}
		catch (Exception e) {
			throw new AnalysisEngineProcessException(e);
		}
		finally {
			closeQuietly(out);
		}
	}

	private void convert(JCas aJCas, PrintWriter aOut)
	{
		int goldTagCount = 0;
		if(filter)
			goldTagCount = annotations.get(DocumentMetaData.get(aJCas).getDocumentId()).size();
		int annotationCount = 0;

		Type neType = JCasUtil.getType(aJCas, NamedEntity.class);
		Feature neValue = neType.getFeatureByBaseName("value");
		StringBuilder sb = new StringBuilder();
		Abstract abstractAnnotation = JCasUtil.select(aJCas, Abstract.class).iterator().next();
		String id = DocumentMetaData.get(aJCas).getDocumentId();

		for (Sentence sentence : select(aJCas, Sentence.class)) {
			HashMap<Token, Row> ctokens = new LinkedHashMap<Token, Row>();

			// Tokens
			List<Token> tokens = selectCovered(Token.class, sentence);

			// Chunks
			IobEncoder encoder = new IobEncoder(aJCas.getCas(), neType, neValue);

			for (int i = 0; i < tokens.size(); i++) {
				Row row = new Row();
				row.id = i+1;
				row.token = tokens.get(i);
				row.ne = encoder.encode(tokens.get(i));
				ctokens.put(row.token, row);
			}

			Row prevRow = null;
			// Write sentence in CONLL 2006 format
			for (Row row : ctokens.values()) {
				String chunk = UNUSED;
				if (writeNamedEntity && (row.ne != null)) {
					chunk = encoder.encode(row.token);

				}
				boolean spacePrev = prevRow != null && prevRow.token.getEnd() == row.token.getBegin()?false:true;

				int begin = row.token.getBegin() >= abstractAnnotation.getBegin()?row.token.getBegin()-abstractAnnotation.getBegin():row.token.getBegin();
				int end = row.token.getBegin() >= abstractAnnotation.getBegin()?row.token.getEnd()-abstractAnnotation.getBegin():row.token.getEnd();
				String typ = row.token.getBegin() >= abstractAnnotation.getBegin()?"A":"T";
				sb.append(String.format(""
						+ "%s%s"
						+ "%s%s"
						+ "%d%s"
						+ "%d%s"
						+ "%s%s"
						+ "%s%s"
						//                		+ "%s%s"
						//                		+ "%s%s"
						+ "%s\n", 
						row.token.getCoveredText(),
						seperator, chunk,
						seperator, begin,
						seperator, end,
						seperator, id,
						seperator, typ,
						seperator, Boolean.toString(spacePrev)
						//                		,seperator, row.token.getLemma().getValue(),
						//                		seperator, row.token.getPos().getPosValue()
						));
				if(chunk.startsWith("B-"))
					annotationCount++;
				prevRow = row;
			}
			if(splitAtSentence)
				sb.append("\n");
		}
		//    	System.out.println(goldTagCount);
		//    	System.out.println(annotationCount);
		//    	System.out.println("---");
		
		if(!splitAtSentence)
			sb.append("\n");

		if((annotationCount!=goldTagCount && filter)||(ids != null &&!ids.contains(DocumentMetaData.get(aJCas).getDocumentId()))){
			//        	System.out.println("error");
			//        	System.out.println(XmlFormatter.getPrettyString(aJCas.getCas()));
		}
		else
			aOut.print(sb.toString());
		if(docCount++%10==0)
			System.out.println(docCount);
	}

	private static final class Row
	{
		int id;
		Token token;
		String ne;
	}
}
