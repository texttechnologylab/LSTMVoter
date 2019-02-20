package org.hucompute.LSTMVoter;

import java.io.IOException;

import org.apache.uima.UIMAException;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.factory.CollectionReaderFactory;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.hucompute.textimager.reader.BioCreativeReader;

import de.tudarmstadt.ukp.dkpro.core.io.conll.Conll2002Writer;
import de.tudarmstadt.ukp.dkpro.core.stanfordnlp.StanfordSegmenter;
import de.tudarmstadt.ukp.dkpro.core.tokit.BreakIteratorSegmenterModified;

public class CreateData {

	public static void main(String[] args) throws UIMAException, IOException {
		String abstractsFile = args[0];
		String annotationsFile = args[1];
		String outputFile = args[2]; 
		CollectionReader textReader = CollectionReaderFactory.createReader(BioCreativeReader.class,
				BioCreativeReader.PARAM_SOURCE_LOCATION,abstractsFile,
				BioCreativeReader.PARAM_ANNOTATION_FILE,annotationsFile,
				BioCreativeReader.PARAM_LANGUAGE,"en",
				BioCreativeReader.PARAM_FILTER_FILES_ONLY_WITH_ANNOTATIONS,true);
		AggregateBuilder builder = new AggregateBuilder();
		builder.add(AnalysisEngineFactory.createEngineDescription(StanfordSegmenter.class));
		builder.add(AnalysisEngineFactory.createEngineDescription(BreakIteratorSegmenterModified.class,BreakIteratorSegmenterModified.PARAM_SPLIT_AT_MINUS,true));
		builder.add(AnalysisEngineFactory.createEngineDescription(Conll2002Writer.class,
				BioCreativeReader.PARAM_ANNOTATION_FILE,annotationsFile,
				Conll2002Writer.PARAM_TARGET_LOCATION,outputFile,
				Conll2002Writer.PARAM_SINGULAR_TARGET,true,
				Conll2002Writer.PARAM_OVERWRITE,true,
				Conll2002Writer.PARAM_SERPERATOR,"\t",
				Conll2002Writer.PARAM_FILTER_FILES_ONLY_WITH_ANNOTATIONS,false,
				Conll2002Writer.PARAM_SPLIT_AT_SENTENCE,true
				));
		SimplePipeline.runPipeline(textReader,builder.createAggregate());
		
	}

}
